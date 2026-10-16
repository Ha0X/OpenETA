import base64
import hashlib
import io
import json

import numpy as np
import pytest
from PIL import Image

from test_codex_robodojo_host import host


def set_frame(remote, pixel, *, action_index=0):
    yy, xx = np.indices((480, 640))
    pixels = np.stack([xx % 256, yy % 256, (xx+yy) % 256], axis=-1).astype('uint8')
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    raw = output.getvalue()
    remote.packet['observation']['cameras'][0]['rgb_base64'] = base64.b64encode(raw).decode()
    remote.packet['receipt'] = {'mark': {'mark_id': 'mark_test', 'camera': 'head',
        'pixel': pixel, 'xyz': [.1, .2, .3], 'source_action_index': action_index, 'depth_m': .4}}
    return raw, pixels


@pytest.mark.parametrize('pixel,origin', [([320, 240], [240, 160]), ([0, 0], [0, 0]),
                                       ([639, 479], [480, 320])])
def test_preview_uses_exact_public_rgb_and_archives_its_derived_image(tmp_path, pixel, origin):
    h, remote = host(tmp_path)
    try:
        raw, pixels = set_frame(remote, pixel)
        result = h.call('mark_point', {'camera': 'head', 'pixel': pixel})
        value = json.loads(result[0].text)
        assert value['error'] is None
        receipt = value['receipt']
        assert receipt['mark'] == remote.packet['receipt']['mark']
        preview = receipt['mark_preview']
        assert preview['original_pixel'] == pixel
        assert preview['crop_box_xyxy'] == [*origin, origin[0]+160, origin[1]+160]
        assert preview['source_rgb_sha256'] == hashlib.sha256(raw).hexdigest()
        assert preview['semantic_identification'] is False
        images = [b for b in result if b.type == 'image']
        assert len(images) == 1
        image_raw = base64.b64decode(images[0].data)
        crop = np.array(Image.open(io.BytesIO(image_raw)))
        assert crop.shape == (160, 160, 3)
        px, py = preview['selected_pixel_in_crop']
        assert np.array_equal(crop[py, px], pixels[pixel[1], pixel[0]])
        assert not h._shown_images  # A crop is not evidence the full image was shown.
        call = h.output/'calls/000001'
        assert (call/preview['artifact']['path']).read_bytes() == image_raw
        assert hashlib.sha256(image_raw).hexdigest() == preview['artifact']['sha256']
        archived = json.loads((call/'response.json').read_text())
        assert [b['sha256'] for b in archived['content'] if b['type'] == 'image'] == [preview['artifact']['sha256']]
        assert (call/'after-000-rgb.png').read_bytes() == raw
        assert base64.b64decode(h.packet['observation']['cameras'][0]['rgb_base64']) == raw
        assert [a[1]['request']['operation'] for a in remote.calls if 'request' in a[1]][-1] == 'mark'
    finally:
        h.close('test_complete')


@pytest.mark.parametrize('problem', ['stale', 'corrupt', 'wrong_dimensions', 'outside'])
def test_unavailable_preview_never_changes_the_mark_or_claims_visual_confirmation(tmp_path, problem):
    h, remote = host(tmp_path)
    try:
        set_frame(remote, [320, 240], action_index=1 if problem == 'stale' else 0)
        frame = remote.packet['observation']['cameras'][0]
        if problem == 'corrupt':
            frame['rgb_base64'] = base64.b64encode(b'not-an-image').decode()
        elif problem == 'wrong_dimensions':
            frame['width'] = 639
        elif problem == 'outside':
            remote.packet['receipt']['mark']['pixel'] = [640, 240]
        result = h.call('mark_point', {'camera': 'head', 'pixel': [320, 240]})
        value = json.loads(result[0].text)
        assert value['error'] is None and value['receipt']['mark_preview']['status'] == 'unavailable'
        assert value['receipt']['mark'] == remote.packet['receipt']['mark']
        assert not any(b.type == 'image' for b in result)
        assert not (h.output/'calls/000001/mark-preview.png').exists()
    finally:
        h.close('test_complete')
