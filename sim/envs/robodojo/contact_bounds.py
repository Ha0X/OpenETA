"""Conservative contact-depth bounds when FCL's queries disagree.

Along any unit axis, shifting one shape until its support intervals separate
is a sufficient separating translation. The smallest such shift among tested
axes is therefore an UPPER bound on penetration depth (not its exact value).
Face normals suffice for a conservative bound; omitting edge-cross axes can
reject a safe contact but cannot underestimate its separating translation.
"""
import numpy as np
from scipy.spatial import ConvexHull

from .scene_geometry import forward_kinematics


def projection_contact_bound(scene, positions, first, second):
    frames=[forward_kinematics(r,q) for r,q in zip(scene.snapshot['robots'],positions)]
    shapes=[];axes=[np.eye(3)]
    for index in (first,second):
        item=scene.items[index]
        if item['approximation']=='sphere':
            center=np.asarray(scene.objects[index].getTranslation(),dtype=float)
            radius=float(scene.geometries[index].radius)
            if not np.isfinite(center).all() or not np.isfinite(radius) or radius<=0:
                raise ValueError('Invalid calibrated sphere contact geometry')
            shapes.append(('sphere',center,radius))
        elif item['approximation']=='convexHull':
            transform=(scene.world_transforms[index] if item['robot_index'] is None
                       else frames[item['robot_index']][item['body']])
            points=scene.vertices[index]@transform[:3,:3].T+transform[:3,3]
            if not np.isfinite(points).all():
                raise ValueError('Invalid convex contact geometry')
            axes.append(ConvexHull(points).equations[:,:3])
            shapes.append(('convex',points,None))
        else:
            raise ValueError('Contact projection requires verified convex or sphere geometry')
    axes=np.concatenate(axes)
    axes=axes/np.linalg.norm(axes,axis=1)[:,None]
    intervals=[]
    for kind,points,radius in shapes:
        if kind=='sphere':
            center=axes@points;intervals.append((center-radius,center+radius))
        else:
            projected=points@axes.T;intervals.append((projected.min(axis=0),projected.max(axis=0)))
    (lo_a,hi_a),(lo_b,hi_b)=intervals
    # Use directional separation distances, not interval intersection length:
    # the latter underestimates penetration for a contained object.
    depth=float(np.min(np.minimum(hi_a-lo_b,hi_b-lo_a)))
    if not np.isfinite(depth):
        raise ValueError('Conservative contact bound unavailable')
    # Negative values conservatively bound penetration; positive values are
    # separation lower bounds. A small numerical guard biases toward overlap.
    return -depth-1e-7
