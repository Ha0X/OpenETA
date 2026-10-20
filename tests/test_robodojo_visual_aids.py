import base64
import copy
import hashlib
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from tools.robodojo_visual_aids import render_tcp_overlay
from tools.harness_config import ConfigRegistry
from tools.harness_evidence import CallEvidence
from mcp.types import ImageContent


def fixture():
    data = BytesIO();Image.new('RGB', (640, 480), '#555555').save(data, format='PNG')
    frame = {'frame_id': 'cam_left_wrist', 'width': 640, 'height': 480,
             'rgb_base64': base64.b64encode(data.getvalue()).decode(),
             'intrinsics': {'fx': 200, 'fy': 200, 'cx': 320, 'cy': 240},
             'extrinsics': {'camera_to_world': np.eye(4).tolist(), 'camera_frame': 'opencv', 'reference_frame': 'world'}}
    tcp = np.eye(4);tcp[2, 3] = 1
    return frame, {'left': tcp.tolist()}


def test_metric_projection_original_unchanged_and_same_pixel_coordinates():
    frame, poses = fixture();before = copy.deepcopy(frame)
    encoded, receipt = render_tcp_overlay(frame, poses)
    assert frame == before
    assert Image.open(BytesIO(base64.b64decode(encoded))).size == (640, 480)
    marker = receipt['markers'][0]
    assert marker['tcp_pixel'] == (320., 240.)
    assert np.allclose(marker['world_axes'][0]['tick_pixels'], [[320+i*2, 240] for i in range(6)])
    assert marker['jaw_axis_pixels'] == [(315., 240.), (325., 240.)]
    assert receipt['source_rgb_sha256'] == hashlib.sha256(base64.b64decode(frame['rgb_base64'])).hexdigest()


def test_camera_pose_rotation_and_translation_are_applied():
    frame, poses = fixture()
    camera = np.eye(4);camera[:3,:3] = [[0,-1,0],[1,0,0],[0,0,1]];camera[0,3] = .1
    frame['extrinsics']['camera_to_world'] = camera.tolist()
    _, receipt = render_tcp_overlay(frame, poses)
    assert np.allclose(receipt['markers'][0]['tcp_pixel'], [320,260])


@pytest.mark.parametrize('damage', ['singular', 'reflection', 'nan', 'behind', 'size', 'frame', 'missing', 'camera', 'focal', 'distortion', 'intrinsic_size', 'unverified'])
def test_invalid_geometry_keeps_fallback_available(damage):
    frame, poses = fixture()
    if damage == 'singular':frame['extrinsics']['camera_to_world'][0][0] = 0
    if damage == 'reflection':frame['extrinsics']['camera_to_world'][0][0] = -1
    if damage == 'nan':poses['left'][0][3] = float('nan')
    if damage == 'behind':poses['left'][2][3] = -1
    if damage == 'size':frame['width'] = 100
    if damage == 'frame':frame['extrinsics']['camera_frame'] = 'opengl'
    if damage == 'missing':del frame['extrinsics']
    if damage == 'camera':frame['frame_id'] = 'unmapped'
    if damage == 'focal':frame['intrinsics']['fx'] = 0
    if damage == 'distortion':frame['intrinsics']['distortion'] = [.2, 0]
    if damage == 'intrinsic_size':frame['intrinsics']['width'] = 800
    if damage == 'unverified':frame['extrinsics']['calibration_status'] = 'unknown'
    encoded, receipt = render_tcp_overlay(frame, poses)
    assert encoded is None and receipt['status'] == 'unavailable' and receipt['reason']
    assert frame['rgb_base64']


def test_non_sensor_annotations_do_not_affect_rendering():
    frame, poses = fixture();first = render_tcp_overlay(frame, poses)
    frame['objects'] = [{'secret_pose': [100, 200, 300]}]
    frame['cloth_mesh'] = 'must not be consulted'
    assert render_tcp_overlay(frame, poses) == first


def test_raw_and_derived_returned_images_are_recoverable(tmp_path):
    frame, poses = fixture();encoded, _ = render_tcp_overlay(frame, poses)
    journal = CallEvidence(tmp_path, 1, 'observe', {}, 'fixture', None)
    journal.complete([ImageContent(type='image', mimeType='image/png', data=data)
                      for data in (frame['rgb_base64'], encoded)], None, received_observation=False)
    response = json.loads((journal.path/'response.json').read_text())
    for block in response['content']:
        raw = (journal.path/block['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == block['sha256']
        Image.open(BytesIO(raw)).verify()


def test_overlay_opt_in_does_not_change_controller_or_physics():
    registry = ConfigRegistry()
    for suffix in ('', '_friction_2x'):
        old = registry.resolve('robodojo_dual_x5_visual_cloth'+suffix+'_v74').value['components']
        new = registry.resolve('robodojo_dual_x5_visual_cloth'+suffix+'_v75').value['components']
        for part in ('benchmark', 'embodiment', 'execution'):
            assert old[part] == new[part]
        assert 'visual_aids' not in old['interface']
        assert new['interface']['visual_aids'] == 'tcp_overlay_v1'
