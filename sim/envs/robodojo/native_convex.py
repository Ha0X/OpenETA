"""Read PhysX's cooked convex parts without changing assets or physical state.

The vertices remain in the source mesh frame. A failed/incomplete query is a
coverage error; combining the parts into one hull would fill container cavities.
"""
import numpy as np


def decode_convex_data(mesh):
    def field(value, name):
        return value[name] if isinstance(value, dict) else getattr(value, name)
    vertices = np.asarray([list(v) for v in field(mesh, 'vertices')], dtype=float)
    raw_indices = np.asarray(field(mesh, 'indices'))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or raw_indices.ndim != 1
            or raw_indices.dtype.kind not in 'iu' or np.any(raw_indices < 0)
            or np.any(raw_indices >= len(vertices))):
        raise ValueError('Invalid native convex vertices or indices')
    triangles = []
    planes = []
    covered = np.zeros(len(raw_indices), dtype=bool)
    for polygon in field(mesh, 'polygons'):
        base, count = field(polygon, 'index_base'), field(polygon, 'num_vertices')
        if (not isinstance(base, (int, np.integer)) or not isinstance(count, (int, np.integer))
                or base < 0 or count < 3 or base+count > len(raw_indices)
                or covered[base:base+count].any()):
            raise ValueError('Invalid native convex polygon range')
        face = raw_indices[base:base+count]
        if len(set(face.tolist())) != count:
            raise ValueError('Repeated native convex polygon vertex')
        covered[base:base+count] = True
        triangles.extend([[int(face[0]), int(face[i]), int(face[i+1])] for i in range(1, count-1)])
        plane = np.asarray(field(polygon, 'plane'), dtype=float)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-12:
            raise ValueError('Invalid native convex plane')
        planes.append(plane/np.linalg.norm(plane[:3]))
    if not covered.all() or not triangles:
        raise ValueError('Incomplete native convex polygon data')
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError
    hull = ConvexHull(vertices)
    if not np.isfinite(hull.volume) or hull.volume <= 0:
        raise ValueError('Degenerate native convex part')
    # PhysX ray tests use the cooked polygon planes. Quantized cooked vertices
    # can differ from those surfaces by tenths of a millimetre; their outer hull
    # is therefore not the same shape. Reconstruct the actual halfspace solid.
    planes = np.asarray(planes)
    try:
        normals = ConvexHull(planes[:, :3])
        if np.max(normals.equations[:, 3]) >= -1e-10:
            raise ValueError('Native convex planes do not bound a closed solid')
        interior = vertices.mean(axis=0)
        if np.max(planes[:, :3] @ interior+planes[:, 3]) >= -1e-10:
            from scipy.optimize import linprog
            solution = linprog([0., 0., 0., -1.],
                A_ub=np.column_stack([planes[:, :3], np.ones(len(planes))]),
                b_ub=-planes[:, 3], bounds=[(None, None)]*4, method='highs')
            if not solution.success or solution.x[3] <= 1e-10:
                raise ValueError('Native convex planes have no strict interior')
            interior = solution.x[:3]
        points = HalfspaceIntersection(planes, interior).intersections
        if not np.isfinite(points).all():
            raise ValueError('Invalid native halfspace intersections')
        exact = ConvexHull(points)
        if exact.volume <= 0 or np.max(points @ planes[:, :3].T+planes[:, 3]) > 1e-8:
            raise ValueError('Native halfspace reconstruction is inconsistent')
    except QhullError as error:
        raise ValueError('Native convex plane reconstruction failed') from error
    return points, exact.simplices.tolist()


def read_convex_parts(mesh_path):
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from pxr import PhysicsSchemaTools, Sdf
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    results = []
    def receive(result, meshes):
        if result != PhysxCollisionRepresentationResult.RESULT_VALID:
            results.append((str(result), []))
            return
        # Copy inside the callback; native mesh-view lifetimes need not outlive it.
        results.append(('valid', [decode_convex_data(mesh) for mesh in meshes]))
    cooking = get_physx_cooking_interface()
    cooking.request_convex_collision_representation(
        omni.usd.get_context().get_stage_id(),
        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(mesh_path)), False, receive)
    if len(results) != 1 or results[0][0] != 'valid' or not results[0][1]:
        status = [item[0] for item in results]
        raise ValueError(f'Native convex decomposition unavailable: {status}')
    return results[0][1]


def world_convex_colliders(parts, mesh_path, collider_path, transform, *, native_approximation="convexDecomposition"):
    """Keep each native part separate, with an explicit refreshable USD source."""
    if native_approximation not in ('convexDecomposition', 'convexHull'):
        raise ValueError('Unsupported native convex representation')
    if native_approximation == 'convexHull' and len(parts) != 1:
        raise ValueError('A native convex hull must return exactly one part')
    if not parts:
        raise ValueError('Native convex decomposition has no parts')
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Invalid native mesh world transform')
    result = []
    for index, (vertices, triangles) in enumerate(parts):
        vertices = np.asarray(vertices, dtype=float)
        world = vertices @ transform[:3,:3].T + transform[:3,3]
        result.append({'id': mesh_path if len(parts)==1 else f'{mesh_path}/__physx_convex_{index:03}',
            'source_prim_path': mesh_path, 'collider_path': collider_path,
            'robot_index': None, 'body': None, 'approximation': 'convexHull',
            'native_approximation': native_approximation,
            'geometry_source': 'physx_cooked_convex_part', 'native_part_index': index,
            'native_part_count': len(parts), 'reference_world_transform': transform.tolist(),
            'vertices': world.tolist(), 'triangles': triangles})
    return result


def require_matching_owner(owners, expected):
    """Absence and conflicting native ownership cannot authorize a binding."""
    if not expected or not owners or set(owners) != {expected}:
        raise ValueError('Native convex collider ownership is unavailable or inconsistent')
    return expected


def verified_convex_owner(env, stage, item):
    """Resolve nested USD bodies from a real PhysX overlap, not ancestry alone."""
    from scipy.spatial.transform import Rotation
    import omni.physx
    from .native_surface import query_delta
    if item.get('geometry_source') != 'physx_cooked_convex_part':
        raise ValueError('Unbound geometry lacks a native cooked convex representation')
    prim=stage.GetPrimAtPath(item['source_prim_path'])
    delta,expected=query_delta(env,stage,prim)
    vertices=np.asarray(item['vertices'],dtype=float)
    low,high=vertices.min(0),vertices.max(0)
    center=(low+high)*.5
    half=(high-low)*.5
    pad=8*np.finfo(np.float32).eps*max(1.,float(abs(center).max()),float(half.max()))
    owners=[]
    def collect(hit):
        if str(hit.collision) in (item['collider_path'],item['source_prim_path']):
            owners.append(str(hit.rigid_body))
        return True
    omni.physx.get_physx_scene_query_interface().overlap_box(
        tuple(half+pad),tuple(delta[:3,:3]@center+delta[:3,3]),
        tuple(Rotation.from_matrix(delta[:3,:3]).as_quat()),collect,False)
    owner=require_matching_owner(owners,expected)
    after,after_owner=query_delta(env,stage,prim)
    if after_owner!=owner or not np.allclose(after,delta,atol=1e-10,rtol=0):
        raise ValueError('Native convex owner moved during binding verification')
    return owner
