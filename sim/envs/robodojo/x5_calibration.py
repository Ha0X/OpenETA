"""Fixed X5 parallel-jaw geometry, verified against deployed robot assets.

The native reported pose is link6. Its +X points towards the fingertips;
fingers slide along +/-Y. The native 0.145 m grip bias lies within the pads
(fingertip mesh ends at link6 x=0.15757 m). No object state is used.
"""
import numpy as np

from .scene_geometry import array, pose_matrix

ROBOT_CONTRACT = 'openeta.x5_parallel_link6_tcp.v1'
TCP_FROM_LINK = np.eye(4)
TCP_FROM_LINK[:3, :3] = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
TCP_FROM_LINK[:3, 3] = [.145, 0., 0.]


def link_to_tcp(transform):
    return array(transform) @ TCP_FROM_LINK


def tcp_to_link(transform):
    return array(transform) @ np.linalg.inv(TCP_FROM_LINK)


def attach_robot_calibration(env, observation):
    from .calibration import SENSOR_CONTRACT
    if getattr(env, '_openeta_sensor_contract', None) != SENSOR_CONTRACT:
        return
    manager = env.robot_manager
    roots = (observation['proprio']['arms'] if 'arms' in observation['proprio']
             else {'primary': observation['proprio']})
    origin = array(manager.scene.env_origins)
    if origin.shape != (1,3):
        raise ValueError('X5 calibrated profile requires one owned world origin')
    for robot, entity in zip(manager.robot_list, manager.robot_key):
        if getattr(robot, 'type', 'target') == 'support':
            continue  # Native environment actors are not Agent X5 effectors.
        role = 'primary' if len(roots)==1 else robot.arm_name.removesuffix('_arm')
        if (role not in roots or robot.robot_name!='x5' or robot.ee_link_name!='link6'
                or robot.ee_type!='gripper' or not np.isclose(robot.gripper_bias,.145,atol=1e-8)):
            raise ValueError('Unverified X5 effector identity')
        entry=roots[role]
        names=list(robot.arm_joints_name)
        if names!=['joint1','joint2','joint3','joint4','joint5','joint6']:
            raise ValueError('Unverified X5 arm joint mapping')
        indices=[entity.joint_names.index(name) for name in names]
        actual=array(entity.data.joint_pos)[0,indices]
        if not np.allclose(actual,entry['joint_positions'],atol=1e-6):
            raise ValueError('Native observation and measured X5 joints disagree')
        physical=array(entity.data.body_link_pose_w)[0,entity.body_names.index('link6')].copy()
        physical[:3]-=origin[0]
        observed=array(entry['ee_pose'])
        reported=pose_matrix(observed[[0,1,2,6,3,4,5]])
        if not np.allclose(reported,pose_matrix(physical),atol=2e-6):
            raise ValueError('Native EE observation does not describe physical link6')
        entry['joint_names']=names
        entry['joint_velocities']=array(entity.data.joint_vel)[0,indices].tolist()
        entry['metadata']={'robot_contract':ROBOT_CONTRACT,'reported_link':'link6',
            'world_origin':'owned_environment','tcp_from_reported':TCP_FROM_LINK.tolist(),
            'tcp_axes':{'approach':'+Z','jaw_opening':'+X'},
            'gripper_geometry':{
                'type':'parallel_jaw', 'frame':'calibrated_tcp',
                'fingertip_extent_along_approach_m':.01257,
                'basis':'deployed_x5_finger_mesh_max_link6_x_minus_native_grip_bias',
                'interpretation':'TCP is inside the finger reach, not at the foremost fingertip. This is robot geometry, not a grasp-depth recommendation.'},
            'tcp_calibration_basis':'native_grip_bias_and_deployed_finger_mesh',
            'gripper_aperture_calibration':'joint_fraction_only'}

        from .gripper_aperture import attach_aperture
        attach_aperture(env, entity, entry, TCP_FROM_LINK)
