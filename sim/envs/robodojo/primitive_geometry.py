"""Exact box geometry for USD Cube colliders; no simulation mutation."""
from itertools import product
import numpy as np


def cube_vertices(size):
    size=float(size)
    if not np.isfinite(size) or size<=0:
        raise ValueError('Cube collider requires finite positive size')
    return np.array(list(product([-size/2,size/2],repeat=3)),dtype=float)


def validate_cube_transform(transform):
    t=np.asarray(transform,dtype=float)
    if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
        raise ValueError('Invalid primitive collider transform')
    scale=np.linalg.norm(t[:3,:3],axis=0)
    if np.any(scale<=0) or not np.allclose(t[:3,:3].T@t[:3,:3],np.diag(scale**2),atol=1e-8,rtol=1e-6):
        raise ValueError('Primitive collider shear or degenerate scale needs cooked-shape verification')
    return t
