import numpy as np
import pytest

from adapter.protocol import CameraFrame
from sim.envs.robodojo.cameras import convert_camera
from tools.harness_robodojo import camera_directory


def test_native_matrix_survives_shared_mcp_serialization_without_authorizing_deprojection():
    raw = {'color': np.zeros((2, 3, 3), dtype=np.uint8),
           'intrinsic_matrix': [[200., 0, 1.5], [0, 200., 1.], [0, 0, 1]],
           'extrinsic_matrix': np.eye(4)}
    frame = convert_camera(raw)
    wire = CameraFrame.from_dict(frame, frame_id='head').to_mcp_dict()
    assert wire['intrinsics']['native_matrix'] == raw['intrinsic_matrix']
    assert wire['extrinsics']['native_matrix'] == np.eye(4).tolist()
    assert wire['width'] == 3 and wire['height'] == 2
    assert wire['depth_base64'] is None
    assert not camera_directory({'cameras': [wire]})['head']['pixel_to_world_available']


@pytest.mark.parametrize('value', [np.eye(3), [[float('nan')]*3]*3, None])
def test_fallback_or_invalid_intrinsics_are_never_promoted(value):
    frame = convert_camera({'intrinsic_matrix': value})
    assert 'fx' not in frame['intrinsics']
    assert frame['intrinsics']['calibration_status'] in {'unavailable','unverified'}


def test_existing_explicit_calibration_is_preserved_without_mutating_input():
    raw = {'intrinsics': {'fx': 200}, 'extrinsics': {'camera_frame': 'opencv'}}
    result = convert_camera(raw)
    result['intrinsics']['fx'] = 1
    assert raw['intrinsics']['fx'] == 200
