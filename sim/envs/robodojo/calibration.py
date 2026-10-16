"""Pinned Isaac camera contract for owned RoboDojo environments.

Reads the actual sensor prim, never its parent mount or object annotations.
The harness world origin is the environment origin, matching native EE poses.
"""
from copy import deepcopy

import numpy as np

SENSOR_CONTRACT = 'openeta.robodojo_isaac_pinhole_rgbd.v1'


def configure_sensors(config):
    result = deepcopy(config)
    vision = result['eval_cfg']['observation']['vision']
    vision.update(depth=True, intrinsic_matrix=False, extrinsic_matrix=False)
    for entry in result['camera']['annotator'].values():
        if entry.get('enabled'):
            entry['optical_depth_capture'] = {'type': 'distance_to_image_plane', 'device': 'cpu'}
    return result


def install_camera_contract():
    from env.camera_manager.camera_manager import CameraManager
    original = CameraManager.create_cameras
    if getattr(original, '_openeta_sensor_contract', None) == SENSOR_CONTRACT:
        return

    def create_cameras(self):
        result = original(self)
        for cameras in self.cameras:
            for camera in cameras:
                if camera.get_lens_distortion_model() != 'pinhole':
                    raise ValueError('Calibrated sensor profile requires pinhole cameras')
                # Isaac only renders square pixels. Upstream explicitly passes
                # False to both aperture setters, leaving head-camera fy wrong.
                camera.set_horizontal_aperture(camera.get_horizontal_aperture(), True)
        return result

    create_cameras._openeta_sensor_contract = SENSOR_CONTRACT
    CameraManager.create_cameras = create_cameras


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def sensor_record(camera, *, rgb_shape, depth_shape, environment_origin):
    width, height = camera.get_resolution()
    if tuple(rgb_shape[:2]) != (height, width) or tuple(depth_shape) != (height, width):
        raise ValueError('Actual RGB, depth and sensor resolutions disagree')
    k = _array(camera.get_intrinsics_matrix())
    view = _array(camera.get_view_matrix_ros())
    origin = _array(environment_origin)
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 1
            or not np.isclose(k[0, 0], k[1, 1], rtol=1e-6)
            or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8
            or origin.shape != (3,) or not np.isfinite(origin).all()):
        raise ValueError('Invalid or non-square-pixel sensor calibration')
    if view.shape != (4, 4) or not np.isfinite(view).all():
        raise ValueError('Invalid actual sensor view transform')
    transform = np.linalg.inv(view)
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=2e-6)):
        raise ValueError('Sensor transform is not rigid')
    transform[:3, 3] -= origin
    return {
        'intrinsics': {'fx': float(k[0, 0]), 'fy': float(k[1, 1]),
                       'cx': float(k[0, 2]), 'cy': float(k[1, 2]),
                       'width': width, 'height': height, 'calibration_status': 'verified',
                       'source': SENSOR_CONTRACT},
        'extrinsics': {'camera_to_world': transform.tolist(), 'camera_frame': 'opencv',
                       'reference_frame': 'world', 'world_origin': 'owned_environment',
                       'depth_convention': 'optical_z_m', 'calibration_status': 'verified',
                       'normalized_from': 'isaac_sensor_view_matrix_ros',
                       'source': SENSOR_CONTRACT},
    }


def attach_calibrated_cameras(env, observation):
    if getattr(env, '_openeta_sensor_contract', None) != SENSOR_CONTRACT:
        return
    manager = env.camera_manager
    if manager.num_envs != 1:
        raise ValueError('Calibrated profile requires one owned environment')
    origins = _array(env.robot_manager.scene.env_origins)
    if origins.shape != (1, 3):
        raise ValueError('Missing owned environment origin')
    names = manager.camera_names[0]
    if set(names) != set(observation['cameras']) or len(set(names)) != len(names):
        raise ValueError('Sensor identity and captured camera directory disagree')
    for index, name in enumerate(names):
        frame = observation['cameras'][name]
        if frame['depth'] is None:
            raise ValueError(f'{name}: calibrated profile requires actual depth')
        frame.update(sensor_record(manager.cameras[0][index], rgb_shape=frame['rgb'].shape,
                                   depth_shape=frame['depth'].shape, environment_origin=origins[0]))
    observation['metadata']['sensor_contract'] = SENSOR_CONTRACT

