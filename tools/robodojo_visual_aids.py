"""Optional image annotations from public camera calibration and robot TCP poses only."""
from __future__ import annotations

import base64
import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw

PROFILE = 'tcp_overlay_v1'
NOTICE = ('Calibrated robot projection, not object detection, contact or grasp evidence. '
          'Markers may be occluded by scene geometry. Same pixels as the original; '
          'use the original image for mark_point. World axis ticks are 1 cm, length 5 cm; '
          'jaw axis shows orientation, not measured finger opening.')


def _rigid(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-4)):
        raise ValueError('invalid_rigid_transform')
    return matrix


def render_tcp_overlay(frame, tcp_poses, *, font=None):
    """Return (PNG base64 or None, receipt). Never infer scene surfaces/attachments.

    Inputs intentionally exclude the rest of the observation and all private geometry.
    Missing/invalid calibration fails back to the unmodified image at the caller.
    """
    record = {'profile': PROFILE, 'camera': frame.get('frame_id'),
              'status': 'unavailable', 'notice': NOTICE, 'markers': []}
    try:
        width, height = frame['width'], frame['height']
        if type(width) is not int or type(height) is not int or not 1 <= width <= 8192 or not 1 <= height <= 8192:
            raise ValueError('invalid_image_size')
        ext = frame['extrinsics']
        if ext.get('camera_frame') != 'opencv' or ext.get('reference_frame') != 'world':
            raise ValueError('unsupported_camera_convention')
        camera_from_world = np.linalg.inv(_rigid(ext['camera_to_world']))
        intr = frame['intrinsics']
        fx, fy, cx, cy = [float(intr[k]) for k in ('fx', 'fy', 'cx', 'cy')]
        if (intr.get('width', width) != width or intr.get('height', height) != height
                or intr.get('calibration_status', 'verified') != 'verified'
                or ext.get('calibration_status', 'verified') != 'verified'):
            raise ValueError('unverified_or_mismatched_calibration')
        if not np.isfinite([fx, fy, cx, cy]).all() or min(fx, fy) <= 0:
            raise ValueError('invalid_intrinsics')
        if any(np.any(np.asarray(intr.get(k, []), dtype=float) != 0) for k in ('distortion', 'distortion_coefficients')):
            raise ValueError('distorted_camera_unsupported')
        raw = base64.b64decode(frame['rgb_base64'], validate=True)
        with Image.open(BytesIO(raw)) as source:
            if source.size != (width, height):
                raise ValueError('calibration_image_size_mismatch')
            canvas = source.convert('RGB')
        draw = ImageDraw.Draw(canvas)
        name = frame['frame_id']
        wrist_role = {'cam_left_wrist': 'left', 'cam_right_wrist': 'right',
                      'left_wrist': 'left', 'right_wrist': 'right'}.get(name)
        if name not in ('head', 'cam_head') and wrist_role is None:
            raise ValueError('unmapped_camera_role')
        poses = dict(tcp_poses)
        if wrist_role == 'left' and set(poses) == {'primary'}:
            wrist_role = 'primary'
        selected = [wrist_role] if wrist_role else list(poses)

        def project(xyz):
            p = camera_from_world @ np.r_[xyz, 1.]
            if p[2] <= 1e-6:
                return None
            uv = np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])
            return tuple(uv.tolist()) if np.isfinite(uv).all() else None

        def visible(p):
            return p is not None and 0 <= p[0] < width and 0 <= p[1] < height

        def label(p, text, color):
            # Keep labels inside the unchanged image bounds.
            box = draw.textbbox((0, 0), text, font=font)
            x = min(max(0, int(p[0]) + 7), max(0, width - box[2] - 2))
            y = min(max(18, int(p[1]) + 7), max(18, height - 14))
            draw.text((x, y), text, fill=color, stroke_width=1, stroke_fill='black', font=font)

        for role in selected:
            pose = _rigid(poses[role])
            origin = pose[:3, 3]
            center = project(origin)
            item = {'role': role, 'tcp_pixel': center, 'visible': visible(center)}
            record['markers'].append(item)
            if not visible(center):
                continue
            color = '#ffce36' if role in ('left', 'primary') else '#ff78f5'
            x, y = center
            draw.line((x-6, y, x+6, y), fill=color, width=2)
            draw.line((x, y-6, x, y+6), fill=color, width=2)
            label(center, role + ' TCP', color)
            if wrist_role:
                axes = []
                for axis, axis_color, direction in zip('XYZ', ('#ff5555', '#55ff88', '#559fff'), np.eye(3)):
                    points = [project(origin + direction * .01 * i) for i in range(6)]
                    for a, b in zip(points, points[1:]):
                        if visible(a) and visible(b):
                            draw.line((*a, *b), fill=axis_color, width=2)
                    for p in points[1:]:
                        if visible(p):
                            draw.ellipse((p[0]-2, p[1]-2, p[0]+2, p[1]+2), fill=axis_color)
                    visible_ticks = [p for p in points[1:] if visible(p)]
                    if visible_ticks:
                        label(visible_ticks[-1], '+' + axis + (' 5cm' if visible(points[-1]) else ' clipped'), axis_color)
                    axes.append({'world_axis': axis, 'tick_pixels': points})
                jaw = [project(origin + pose[:3, 0] * d) for d in (-.025, .025)]
                if all(visible(p) for p in jaw):
                    draw.line((*jaw[0], *jaw[1]), fill='#ffffff', width=1)
                    label(jaw[0], 'jaw axis', '#ffffff')
                item.update(world_axes=axes, jaw_axis_pixels=jaw)
        if not any(m['visible'] for m in record['markers']):
            raise ValueError('no_tcp_in_view')
        draw.rectangle((0, 0, width, 17), fill='black')
        draw.text((3, 2), 'ROBOT PROJECTION | not grasp evidence | original pixels', fill='white', font=font)
        stream = BytesIO()
        canvas.save(stream, format='PNG')
        data = stream.getvalue()
        record.update(status='available', source_rgb_sha256=hashlib.sha256(raw).hexdigest(),
                      overlay_sha256=hashlib.sha256(data).hexdigest())
        return base64.b64encode(data).decode(), record
    except (KeyError, ValueError, TypeError, OSError, np.linalg.LinAlgError) as exc:
        record['reason'] = str(exc)
        return None, record
