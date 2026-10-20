"""Opt-in lossless model-image encoding; public sensor archives stay untouched."""
from __future__ import annotations

import base64
import io

POLICY = 'lossless_webp_if_smaller_v1'
MAX_PIXELS = 4_000_000


def encode_image(block):
    """Only replace a PNG after decoding proves exact RGBA pixel equality."""
    record = {'source_mime': block.mimeType, 'delivered_mime': block.mimeType,
              'pixels_unchanged': True, 'status': 'original'}
    if block.mimeType != 'image/png':
        return block, {**record, 'reason': 'source_format_preserved'}
    try:
        from PIL import Image, features
        if not features.check('webp'):
            return block, {**record, 'reason': 'encoder_unavailable'}
        raw = base64.b64decode(block.data, validate=True)
        record['source_bytes'] = record['delivered_bytes'] = len(raw)
        with Image.open(io.BytesIO(raw)) as source:
            # Preserve color interpretation and animation, not just pixel values.
            if (source.format != 'PNG' or source.mode not in ('RGB', 'RGBA')
                    or getattr(source, 'n_frames', 1) != 1
                    or source.width * source.height > MAX_PIXELS
                    or any(k in source.info for k in ('icc_profile', 'exif', 'gamma', 'srgb', 'chromaticity', 'transparency'))):
                return block, {**record, 'reason': 'source_semantics_preserved'}
            source.load()
            buffer = io.BytesIO()
            source.save(buffer, format='WEBP', lossless=True, exact=True, method=4)
            encoded = buffer.getvalue()
            if len(encoded) >= len(raw):
                return block, {**record, 'reason': 'not_smaller'}
            with Image.open(io.BytesIO(encoded)) as decoded:
                if (decoded.size != source.size
                        or decoded.convert('RGBA').tobytes() != source.convert('RGBA').tobytes()):
                    return block, {**record, 'reason': 'pixel_verification_failed'}
        return block.model_copy(update={'data': base64.b64encode(encoded).decode(), 'mimeType': 'image/webp'}), {
            **record, 'status': 'lossless_encoded', 'delivered_mime': 'image/webp',
            'delivered_bytes': len(encoded)}
    except (ImportError, OSError, ValueError, RuntimeError):
        # A transport optimization must never remove an already available view.
        return block, {**record, 'reason': 'encoding_unavailable_or_invalid'}


def encode_content(content):
    output, records = [], []
    for index, block in enumerate(content):
        if block.type == 'image':
            block, record = encode_image(block)
            records.append({'content_index': index, **record})
        output.append(block)
    return output, {'policy': POLICY, 'views_unchanged': True,
                    'pixel_coordinates_unchanged': True, 'images': records}
