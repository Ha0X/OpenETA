"""Private outer covers from a complete, bounded native surface oracle.

This module does not certify a PhysX backend by itself. Its oracle must report
EVERY surface intersection with an axis-aligned box in the declared bounds.
Unknown results and budgets fail closed. Closed interior components are filled;
exterior-connected slots remain free. The resulting cells conservatively cover
the queried surface, not an unvalidated SDF/contact surface or future deformation.
"""
import base64
import math
import time
from itertools import product

import numpy as np


class CoverUnavailable(ValueError):
    pass


def surface_cover(bounds, intersects, *, resolution=.0005, max_cells=16777216,
                  max_queries=1000000, timeout_s=120.):
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not math.isfinite(resolution)
            or resolution <= 0 or not math.isfinite(timeout_s) or timeout_s <= 0):
        raise CoverUnavailable('Invalid native surface cover domain')
    # One guaranteed empty boundary layer. A hit there is an inconsistent
    # domain/oracle, never clipped away to manufacture complete coverage.
    low = np.floor(bounds[0]/resolution).astype(np.int64)-2
    high = np.ceil(bounds[1]/resolution).astype(np.int64)+2
    shape = high-low
    if np.any(low < -32768) or np.any(high > 32768):
        raise CoverUnavailable('Native cover exceeds OctoMap coordinate range')
    if math.prod(map(int, shape)) > max_cells:
        raise CoverUnavailable('Native cover cell budget; resolution was not enlarged')
    occupied = np.zeros(tuple(shape), dtype=bool)
    deadline = time.monotonic()+timeout_s
    stack = [(np.zeros(3, dtype=int), shape.copy())]
    queries = 0
    max_pad = 0.
    while stack:
        if queries >= max_queries or time.monotonic() > deadline:
            raise CoverUnavailable(f'Native cover query budget; queries={queries}, occupied={int(occupied.sum())}, pending={len(stack)}, time_remaining_s={deadline-time.monotonic():.3f}; incomplete cover discarded')
        a, b = stack.pop()
        center = (low+(a+b)*.5)*resolution
        half = (b-a)*(.5*resolution)
        # Guard float32 scene-query conversion outwards, including cell borders.
        pad = 8*np.finfo(np.float32).eps*max(1.,float(abs(center).max()),float(half.max()))
        max_pad = max(max_pad,pad)
        hit = intersects(center, half+pad)
        queries += 1
        if not isinstance(hit, (bool, np.bool_)):
            raise CoverUnavailable('Native cover query returned unknown')
        if not hit:
            continue
        widths = b-a
        if np.all(widths == 1):
            occupied[tuple(a)] = True
            continue
        axis = int(np.argmax(widths))
        middle = int((a[axis]+b[axis])//2)
        left_b, right_a = b.copy(), a.copy()
        left_b[axis] = right_a[axis] = middle
        stack.extend([(a,left_b),(right_a,b)])
    if not occupied.any():
        raise CoverUnavailable('Native surface oracle returned no geometry')
    if any(np.take(occupied, k, axis=axis).any() for axis in range(3) for k in (0,-1)):
        raise CoverUnavailable('Native surface extends outside declared bounds')
    from scipy.ndimage import binary_fill_holes
    filled = binary_fill_holes(occupied)
    keys = np.argwhere(filled)+low
    return {'indices':keys,'resolution_m':float(resolution),'queries':queries,
            'surface_cells':int(occupied.sum()),'occupied_cells':len(keys),
            'interior_policy':'fill_components_not_connected_to_exterior',
            'surface_cover_error_m':float(math.sqrt(3)*resolution+max_pad)}


def encode_octomap(indices):
    """OctoMap binary node stream, depth 16; explicitly free absent children.

    Child index uses x + 2*y + 4*z bits. Two bits encode each child: 1 free,
    2 occupied, 3 inner node. No native pointers or opaque pickle payloads.
    """
    indices = np.asarray(indices)
    if (indices.ndim != 2 or indices.shape[1] != 3 or not len(indices)
            or indices.dtype.kind not in 'iu' or np.any(indices < -32768)
            or np.any(indices >= 32768)):
        raise CoverUnavailable('Invalid occupied voxel indices')
    keys = np.unique(indices.astype(np.int64), axis=0)+32768

    def node(points, bit):
        if not len(points):
            return 1, b''
        if bit < 0 or len(points) == 8**(bit+1):
            return 2, b''
        children = ((points[:,0]>>bit)&1)+2*((points[:,1]>>bit)&1)+4*((points[:,2]>>bit)&1)
        values = [node(points[children==i],bit-1) for i in range(8)]
        code = sum(v[0] << (2*i) for i,v in enumerate(values))
        return 3, code.to_bytes(2,'little')+b''.join(v[1] for v in values if v[0]==3)
    kind, stream = node(keys,15)
    if kind != 3:
        raise CoverUnavailable('Unexpected full root voxel cover')
    return stream


def cover_collider(cover, *, mesh_path, collider_path, reference_transform, native_approximation):
    keys = np.asarray(cover['indices'])
    resolution = cover['resolution_m']
    low, high = keys.min(axis=0)*resolution, (keys.max(axis=0)+1)*resolution
    return {'id':mesh_path,'collider_path':collider_path,'source_prim_path':mesh_path,
            'robot_index':None,'body':None,'approximation':'nativeQueryVoxels',
            'native_approximation':native_approximation,
            'geometry_source':'native_surface_query_outer_voxel_cover',
            'vertices':np.array(list(product(*zip(low,high)))).tolist(),'triangles':[],
            'reference_world_transform':np.asarray(reference_transform).tolist(),
            'voxel_resolution_m':resolution,
            'octomap_binary':base64.b64encode(encode_octomap(keys)).decode('ascii'),
            'cover_evidence':{k:v for k,v in cover.items() if k!='indices'}}


def fcl_octree(resolution, stream):
    """Honor the platform C++ char ABI (signed x86, unsigned ARM builds)."""
    import fcl
    if not math.isfinite(resolution) or resolution <= 0 or not isinstance(stream, bytes) or not stream:
        raise CoverUnavailable('Invalid native voxel FCL payload')
    try:
        return fcl.OcTree(resolution, stream)
    except OverflowError:
        return fcl.OcTree(resolution, np.frombuffer(stream, dtype=np.int8).tolist())


def separation_translation_bound(scene, first, second):
    """Verified separating translation upper bound for a voxel-union contact.

    Maximum leaf penetration is NOT a penetration bound for a union. Instead
    test a translated copy of one complete object against the other complete
    object. Only an actually disjoint endpoint certifies an upper bound; the
    original objects and simulation are never moved. This is not exact depth.
    """
    fcl=scene.fcl
    object_a,object_b=scene.objects[first],scene.objects[second]
    rotation=np.asarray(object_a.getRotation(),dtype=float)
    origin=np.asarray(object_a.getTranslation(),dtype=float)
    probe=fcl.CollisionObject(scene.geometries[first],fcl.Transform(rotation,origin))
    axes=[*np.eye(3),*rotation.T,*np.asarray(object_b.getRotation()).T]
    # Directions are a heuristic for tightness only. Every chosen endpoint is
    # checked against the full union, so missing a direction cannot relax it.
    best=math.inf
    for direction in axes:
        for sign in (-1,1):
            direction=sign*np.asarray(direction)
            for distance in (.00025,.0005,.000999,.002,.004,.008,.016,.032,.064,.128,.256,.512,1.):
                if distance>=best:break
                probe.setTransform(fcl.Transform(rotation,origin+direction*distance))
                result=fcl.CollisionResult()
                fcl.collide(probe,object_b,fcl.CollisionRequest(),result)
                if not result.is_collision:
                    # Keep an explicitly collision-free upper endpoint even
                    # when cavities make occupancy nonmonotone along the ray.
                    low,high=0.,distance
                    for _ in range(12):
                        middle=(low+high)*.5
                        probe.setTransform(fcl.Transform(rotation,origin+direction*middle))
                        result=fcl.CollisionResult()
                        fcl.collide(probe,object_b,fcl.CollisionRequest(),result)
                        if result.is_collision:low=middle
                        else:high=middle
                    best=min(best,high)
                    break
    if not math.isfinite(best):raise CoverUnavailable('Voxel contact separating bound unavailable')
    return best+1e-7


def native_query_domain(cooked_bounds, points, transform, *, sdf_margin=0.):
    """Query domain contains BOTH the source mesh and the native cooked bounds.

    Property-query cooked bounds can exclude source triangles. They are not a
    sufficient domain on their own. Domain expansion changes only where the
    oracle is queried; occupied geometry still comes exclusively from hits.
    """
    bounds=np.asarray(cooked_bounds,dtype=float)
    points=np.asarray(points,dtype=float)
    transform=np.asarray(transform,dtype=float)
    if (bounds.shape!=(2,3) or points.ndim!=2 or points.shape[1]!=3 or not len(points)
            or transform.shape!=(4,4) or not np.isfinite(bounds).all()
            or not np.isfinite(points).all() or not np.isfinite(transform).all()
            or not np.allclose(transform[3],[0,0,0,1]) or np.any(bounds[1]<bounds[0])
            or not math.isfinite(sdf_margin) or sdf_margin<0):
        raise CoverUnavailable('Invalid native/source query domain')
    low=np.minimum(bounds[0],points.min(0));high=np.maximum(bounds[1],points.max(0))
    corners=np.array(list(product(*zip(low,high))))@transform[:3,:3].T+transform[:3,3]
    world=np.array([corners.min(0),corners.max(0)])
    # The schema defines margin as a fraction of mesh bounding-box diagonal.
    # A world AABB diagonal conservatively includes rotation/nonuniform scale.
    padding=float(sdf_margin*np.linalg.norm(world[1]-world[0]))
    return world+np.array([[-padding]*3,[padding]*3])
