from types import SimpleNamespace

import numpy as np
import pytest

from adapter.protocol import CameraFrame
from sim.envs.robodojo.calibration import sensor_record, configure_sensors
from tools.harness_robodojo import camera_directory


def sensor(fy=200):
    # Optical +Z faces world +X, and an environment origin is translated.
    rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    t=np.eye(4);t[:3,:3]=rotation;t[:3,3]=[10.,20.,3.]
    return SimpleNamespace(get_resolution=lambda:(640,480),
        get_intrinsics_matrix=lambda:np.array([[200.,0.,320.],[0.,fy,240.],[0.,0.,1.]]),
        get_view_matrix_ros=lambda:np.linalg.inv(t))


def test_actual_optical_sensor_and_environment_origin_survive_wire():
    record=sensor_record(sensor(),rgb_shape=(480,640,3),depth_shape=(480,640),environment_origin=[9.,18.,0.])
    t=np.array(record['extrinsics']['camera_to_world'])
    # Center pixel at optical depth 2 is world [3,2,3] in the owned frame.
    assert np.allclose(t@np.array([0.,0.,2.,1.]),[3.,2.,3.,1.])
    wire=CameraFrame.from_dict({**record,'rgb':np.zeros((480,640,3),np.uint8),
        'depth':np.ones((480,640),np.float32)},frame_id='head').to_mcp_dict()
    assert camera_directory({'cameras':[wire]})['head']['pixel_to_world_available']
    assert wire['extrinsics']['depth_convention']=='optical_z_m'


def test_known_head_aperture_bug_is_not_accepted_as_calibration():
    with pytest.raises(ValueError,match='square-pixel'):
        sensor_record(sensor(fy=230),rgb_shape=(480,640,3),depth_shape=(480,640),environment_origin=[0,0,0])


@pytest.mark.parametrize('rgb,depth', [((256,256,3),(480,640)),((480,640,3),(640,480))])
def test_image_size_mismatch_does_not_silently_change_projection(rgb,depth):
    with pytest.raises(ValueError,match='resolutions'):
        sensor_record(sensor(),rgb_shape=rgb,depth_shape=depth,environment_origin=[0,0,0])


def test_sensor_profile_enables_all_real_camera_depth_paths_without_mutating_source():
    source={'eval_cfg':{'observation':{'vision':{'depth':False}}},
        'camera':{'annotator':{'common':{'enabled':True},'cam_head':{'enabled':True},'disabled':{'enabled':False}}}}
    result=configure_sensors(source)
    assert not source['eval_cfg']['observation']['vision']['depth']
    assert result['eval_cfg']['observation']['vision']['depth']
    assert result['camera']['annotator']['common']['optical_depth_capture']['type']=='distance_to_image_plane'
    assert 'optical_depth_capture' in result['camera']['annotator']['cam_head']
    assert 'optical_depth_capture' not in result['camera']['annotator']['disabled']


def test_x5_tcp_axes_and_offset_are_not_panda_conventions():
    from sim.envs.robodojo.x5_calibration import link_to_tcp,tcp_to_link
    link=np.eye(4);link[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];link[:3,3]=[1.,2.,3.]
    tcp=link_to_tcp(link)
    assert np.allclose(tcp[:3,3],[1.,2.145,3.])
    assert np.allclose(tcp[:3,2],[0.,1.,0.])
    assert np.allclose(tcp[:3,0],[-1.,0.,0.])
    assert np.allclose(tcp_to_link(tcp),link)
