import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from mcp.types import ImageContent, TextContent

from tools.robodojo_image_transport import encode_image, encode_content, POLICY
from tools.harness_evidence import CallEvidence
from tools.harness_config import ConfigRegistry, ConfigError, ResolvedConfig, canonical


def texture(alpha=False):
    rng = np.random.default_rng(317)
    grey = rng.integers(0, 256, (96, 96), dtype=np.uint8)
    pixels = np.repeat(grey[:, :, None], 3, axis=2)
    if alpha:
        pixels = np.concatenate([pixels, rng.integers(0, 256, (96, 96, 1), dtype=np.uint8)], axis=2)
    image = Image.fromarray(pixels)
    buffer = io.BytesIO(); image.save(buffer, format='PNG')
    return image, ImageContent(type='image', mimeType='image/png', data=base64.b64encode(buffer.getvalue()).decode())


@pytest.mark.parametrize('alpha', [False, True])
def test_lossless_pixels_including_transparent_rgb_and_coordinates(alpha):
    source, block = texture(alpha)
    original = block.model_dump()
    encoded, report = encode_image(block)
    assert encoded.mimeType == 'image/webp'
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded.data)))
    assert decoded.size == source.size
    assert decoded.convert('RGBA').tobytes() == source.convert('RGBA').tobytes()
    assert report['delivered_bytes'] < report['source_bytes']
    assert block.model_dump() == original


def test_encoder_failure_keeps_original_view(monkeypatch):
    _, block = texture()
    def fail(*args, **kwargs):raise OSError('encoder unavailable')
    monkeypatch.setattr(Image.Image, 'save', fail)
    encoded, report = encode_image(block)
    assert encoded is block
    assert report['reason'] == 'encoding_unavailable_or_invalid'


def test_larger_encoding_keeps_original(monkeypatch):
    _, block = texture()
    save = Image.Image.save
    def padded(self, fp, *args, **kwargs):
        save(self, fp, *args, **kwargs)
        fp.write(b'padding' * 10000)
    monkeypatch.setattr(Image.Image, 'save', padded)
    encoded, report = encode_image(block)
    assert encoded is block and report['reason'] == 'not_smaller'


def test_invalid_pixel_conversion_is_not_delivered(monkeypatch):
    _, block = texture()
    save = Image.Image.save
    def changed(self, fp, *args, **kwargs):
        return save(Image.new('RGB', self.size, 'red'), fp, *args, **kwargs)
    monkeypatch.setattr(Image.Image, 'save', changed)
    encoded, report = encode_image(block)
    assert encoded is block and report['reason'] == 'pixel_verification_failed'


def test_color_profile_semantics_and_unrelated_formats_preserved():
    from PIL.PngImagePlugin import PngInfo
    source, _ = texture();info=PngInfo();info.add(b'gAMA', (45455).to_bytes(4,'big'))
    buffer=io.BytesIO();source.save(buffer,format='PNG',pnginfo=info)
    block=ImageContent(type='image',mimeType='image/png',data=base64.b64encode(buffer.getvalue()).decode())
    encoded, report=encode_image(block)
    assert encoded is block and report['reason']=='source_semantics_preserved'
    jpeg=block.model_copy(update={'mimeType':'image/jpeg'})
    assert encode_image(jpeg)[0] is jpeg


def test_exact_returned_image_is_archived_with_truthful_extension(tmp_path):
    _, block=texture();text=TextContent(type='text',text='original camera coordinates')
    encoded, report=encode_content([text,block])
    assert encoded[0] is text and report['views_unchanged']
    evidence=CallEvidence(tmp_path,1,'observe',{},'test',None)
    evidence.complete(encoded,None,received_observation=False)
    row=json.loads((evidence.path/'response.json').read_text())['content'][1]
    assert row['mimeType']=='image/webp' and row['path'].endswith('.webp')
    assert (evidence.path/row['path']).read_bytes()==base64.b64decode(encoded[1].data)


def test_host_opt_in_retains_original_and_tcp_view_and_png_sensor_evidence(tmp_path):
    from tests.test_codex_robodojo_host import host
    h,r=host(tmp_path)
    try:
        value=h.config.value
        value['components']['interface'].update(image_encoding=POLICY,visual_aids='tcp_overlay_v1')
        h.config=ResolvedConfig(canonical(value))
        source,block=texture()
        frame=r.packet['observation']['cameras'][0]
        frame.update(rgb_base64=block.data,width=96,height=96)
        frame['intrinsics'].update(cx=48,cy=48,width=96,height=96)
        camera=np.eye(4);camera[:3,3]=np.asarray(r.packet['control']['tcp_poses']['primary'])[:3,3]-[0,0,1]
        frame['extrinsics']['camera_to_world']=camera.tolist()
        result=h.call('observe',{})
        images=[b for b in result if b.type=='image']
        assert len(images)==2
        assert images[0].mimeType=='image/webp'
        assert Image.open(io.BytesIO(base64.b64decode(images[0].data))).tobytes()==source.tobytes()
        body=json.loads(result[0].text)
        assert body['image_transport']['policy']==POLICY
        assert len(body['image_transport']['images'])==2
        assert (h.output/'calls/000001/after-000-rgb.png').read_bytes()==base64.b64decode(block.data)
    finally:h.close('test')


def test_only_new_preset_opts_in_and_is_executable():
    registry=ConfigRegistry()
    config=registry.resolve('robodojo_dual_x5_conveyor_lossless_v86')
    config.require_executable()
    assert config.value['components']['interface']['image_encoding']==POLICY
    for name in ['robodojo_dual_x5_conveyor_image_v85','libero_panda_v1']:
        baseline=registry.resolve(name);baseline.require_executable()
        assert 'image_encoding' not in baseline.value['components']['interface']


def test_invalid_codec_and_libero_opt_in_rejected(tmp_path):
    from tests.test_harness_config import catalog_copy
    def invalid(c,p):c['components']['interface']['robodojo_dual_atomic_visual_v75']['image_encoding']='jpeg_lossy'
    with pytest.raises(ConfigError,match='robodojo_dual_atomic_visual_v75'):catalog_copy(tmp_path,invalid)
    def libero(c,p):c['components']['interface']['libero_atomic_v1']['image_encoding']=POLICY
    with pytest.raises(ConfigError,match='libero_atomic_v1'):catalog_copy(tmp_path,libero)
