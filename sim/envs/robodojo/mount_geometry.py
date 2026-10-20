"""Pinned fixed-table variants checked against the active native layout.

Only support geometry is read here. No task object labels or target poses are
exported, and matching a layout does not exempt any robot link except its base.
"""
from itertools import product
import numpy as np


TABLE_VARIANTS = (
    ('wide_centered_v1', [1.4, 1.1, .05], [0., -.05, .74]),
    ('narrow_offset_v1', [1., 1.1, .05], [.09, -.05, .74]),
)


def calibrated_table_geometry(config):
    try:
        scale = np.asarray(config['scale'], dtype=float)
        position = np.asarray(config['default_pos'], dtype=float)
        orientation = np.asarray(config['default_ori'], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Active native tabletop layout is incomplete') from exc
    if (scale.shape != (3,) or position.shape != (3,) or orientation.shape != (4,)
            or not all(np.isfinite(x).all() for x in (scale, position, orientation))
            or not (np.allclose(orientation, [1, 0, 0, 0], atol=1e-8, rtol=0)
                    or np.allclose(orientation, [-1, 0, 0, 0], atol=1e-8, rtol=0))):
        raise ValueError('Active native tabletop layout is not calibrated')
    for name, expected_scale, expected_position in TABLE_VARIANTS:
        if (np.allclose(scale, expected_scale, atol=1e-8, rtol=0)
                and np.allclose(position, expected_position, atol=1e-8, rtol=0)):
            corners = np.array(list(product(*zip(position-scale/2, position+scale/2))))
            return name, corners
    raise ValueError('Active native tabletop layout is not a calibrated variant')
