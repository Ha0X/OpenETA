"""Preserve native calibration evidence without guessing optical conventions.

The shared CameraFrame protocol accepts dictionaries, not raw matrix lists.
Native RoboDojo extrinsics describe a scene camera transform whose optical
axes, sensor offset and environment origin must still be verified on deployment.
"""
from copy import deepcopy

import numpy as np


def calibration_record(value, *, kind):
    if isinstance(value, dict):
        return deepcopy(value)
    shape = (3, 3) if kind == 'intrinsics' else (4, 4)
    try:
        matrix = np.asarray(value)
        valid = matrix.shape == shape and matrix.dtype.kind in 'fiu' and np.isfinite(matrix).all()
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return {'calibration_status': 'unavailable', 'reason': 'native_matrix_missing_or_invalid'}
    # Do not emit fx/fy or a canonical camera_to_world transform from this
    # evidence alone. In particular, the native K can be a fallback identity.
    return {'native_matrix': matrix.tolist(), 'source': f'robodojo.camera_manager.{kind}',
            'calibration_status': 'unverified',
            'reason': 'optical_axes_sensor_offset_origin_and_depth_not_validated'}


def convert_camera(frame):
    depth = frame.get('depth')
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        depth = np.where(np.isfinite(depth) & (depth >= 0), depth, 0).astype(np.float32)
    return {'rgb': np.asarray(frame.get('color', frame.get('rgb', [])), dtype=np.uint8),
            'depth': depth,
            'intrinsics': calibration_record(frame.get('intrinsic_matrix', frame.get('intrinsics', {})), kind='intrinsics'),
            'extrinsics': calibration_record(frame.get('extrinsic_matrix', frame.get('extrinsics', {})), kind='extrinsics')}
