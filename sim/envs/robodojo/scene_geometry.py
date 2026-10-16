"""Operator-private USD collision and articulation snapshot.

No simulator state is changed. Unsupported collision geometry is an explicit
coverage error, never silently treated as empty space. World decompositions
use native cooked parts. Robot decompositions retain conservative enclosing
hulls because the current PhysX query cannot parse their Xform-owned meshes.
"""
import numpy as np
from itertools import product


def collision_envelope(points, triangles, approximation):
    if approximation == 'boundingSphere':
        center = (points.min(axis=0)+points.max(axis=0))/2
        distances = np.linalg.norm(points-center, axis=1)
        radius = float(distances.max())
        if (radius <= 0 or not np.isfinite(radius) or
                distances.min() < radius*(1-1e-5) or
                not np.allclose(points.max(axis=0)-center, radius, rtol=1e-5, atol=1e-9)):
            raise ValueError('boundingSphere requires a verified spherical mesh; arbitrary fitted spheres are unsupported')
        radius += 1e-9
        return center+np.array(list(product([-radius,radius], repeat=3))), [], 'sphere'
    if approximation == 'boundingCube':
        points = np.array(list(product(*zip(points.min(axis=0), points.max(axis=0)))))
    if approximation in {'boundingCube', 'convexHull', 'convexDecomposition'}:
        from scipy.spatial import ConvexHull, QhullError
        try:
            hull = ConvexHull(points)
        except QhullError as error:
            raise ValueError('Degenerate convex collision mesh requires native cooked geometry; source vertices are insufficient') from error
        kept = hull.vertices
        mapping = {int(old): new for new, old in enumerate(kept)}
        triangles = [[mapping[int(i)] for i in face] for face in hull.simplices]
        return points[kept], triangles, 'convexHull'
    if approximation != 'none':
        raise ValueError(f'Unsupported native collision approximation: {approximation}')
    return points, triangles, approximation


def transform_vertices(vertices, transform, approximation):
    """Sphere vertices encode its conservative local AABB, not a box mesh."""
    if approximation != 'sphere':
        return vertices @ transform[:3,:3].T+transform[:3,3]
    scale = np.linalg.norm(transform[:3,:3],axis=0)
    if (not np.allclose(scale,scale[0],rtol=1e-6) or scale[0] <= 0 or
            not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3)*scale[0]**2,rtol=1e-6,atol=1e-10)):
        raise ValueError('Native sphere requires uniform scale without shear')
    center=(vertices.min(axis=0)+vertices.max(axis=0))/2
    radius=float((vertices.max(axis=0)-vertices.min(axis=0))[0]/2)*scale[0]
    center=transform[:3,:3]@center+transform[:3,3]
    return center+np.array(list(product([-radius,radius],repeat=3)))


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def quaternion_matrix(q):
    w, x, y, z = array(q) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def pose_matrix(pose):
    pose = array(pose)
    result = np.eye(4)
    result[:3, :3] = quaternion_matrix(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def joint_frame(joint, side):
    q = getattr(joint, f'GetLocalRot{side}Attr')().Get()
    p = getattr(joint, f'GetLocalPos{side}Attr')().Get()
    return pose_matrix([*p, q.GetReal(), *q.GetImaginary()])


def snapshot_scene(env):
    from pxr import Usd, UsdGeom, UsdPhysics
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    result = {'schema_version': 'openeta.robodojo_collision_snapshot.v1',
              'geometry_model': 'usd_geometry_with_native_world_convex_decomposition',
              'robots': [], 'colliders': [], 'unsupported': [], 'filtered_pairs': []}
    roots = {}
    for index, (robot, entity) in enumerate(zip(env.robot_manager.robot_list, env.robot_manager.robot_key)):
        root = str(entity.cfg.prim_path).replace('env_.*', 'env_0')
        if not stage.GetPrimAtPath(root).IsValid():
            raise ValueError(f'Cannot resolve physical robot root {root}')
        if root in roots:
            raise ValueError('Coupled articulation needs a joint ownership adapter')
        roots[root] = index
        poses = array(entity.data.body_link_pose_w)
        if poses.shape != (1, len(entity.body_names), 7):
            raise ValueError('Expected one owned physical articulation')
        frames = {name: pose_matrix(poses[0, i]).tolist() for i, name in enumerate(entity.body_names)}
        result['robots'].append({'root': root, 'role': 'primary' if len(env.robot_manager.robot_list)==1
                                 else robot.arm_name.removesuffix('_arm'),
                                 'joint_names': list(entity.joint_names),
                                 'joint_positions': array(entity.data.joint_pos)[0].tolist(),
                                 'joint_limits': array(entity.data.joint_pos_limits)[0].tolist(),
                                 'fixed_base': bool(entity.is_fixed_base),
                                 'body_frames': frames, 'base_link': robot.base_link,
                                 'joints': [], 'arm_joint_names': list(robot.arm_joints_name),
                                 'control_owner': getattr(robot, 'type', 'target'),
                                 'tool_link': 'panda_hand' if 'panda_hand' in frames else 'link6',
                                 'finger_links': (['panda_leftfinger', 'panda_rightfinger']
                                                  if 'panda_hand' in frames else ['link7', 'link8'])})

    def owner(path):
        for root, index in roots.items():
            if path.startswith(root+'/'):
                body = path[len(root)+1:].split('/')[0]
                if body in result['robots'][index]['body_frames']:
                    return index, body
        return None, None

    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                result['filtered_pairs'].append([path, str(other)])
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) != 1 or len(b) != 1:
                continue  # fixed world root is supplied by the measured base
            ia, ba = owner(str(a[0])); ib, bb = owner(str(b[0]))
            if ia is None or ia != ib:
                if ia is not None or ib is not None:
                    result['unsupported'].append({'path':path,'reason':'external_articulation_joint'})
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                kind, axis = 'revolute', UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                kind, axis = 'prismatic', UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get()
            elif prim.IsA(UsdPhysics.FixedJoint):
                kind, axis = 'fixed', 'X'
            else:
                result['unsupported'].append({'path':path,'reason':'unsupported_joint'})
                continue
            result['robots'][ia]['joints'].append({'name':prim.GetName(), 'parent':ba, 'child':bb,
                'kind':kind, 'axis':axis, 'parent_frame':joint_frame(joint,0).tolist(),
                'child_frame':joint_frame(joint,1).tolist(),
                'collision_enabled':bool(joint.GetCollisionEnabledAttr().Get())})
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        index, body = owner(path)
        approximation = prim.GetAttribute('physics:approximation').Get() or 'none'
        meshes = [p for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        if prim.IsA(UsdGeom.Cube):
            from .primitive_geometry import cube_vertices, validate_cube_transform
            points = cube_vertices(UsdGeom.Cube(prim).GetSizeAttr().Get())
            transform = validate_cube_transform(np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T)
            if index is not None:
                transform = np.linalg.inv(array(result['robots'][index]['body_frames'][body])) @ transform
            vertices = points @ transform[:3,:3].T + transform[:3,3]
            result['colliders'].append({'id':path, 'collider_path':path,
                'robot_index':index, 'body':body, 'approximation':'convexHull',
                'native_approximation':'analyticCube',
                'reference_world_transform':transform.tolist() if index is None else None,
                'vertices':vertices.tolist(), 'triangles':[]})
            continue
        if not meshes:
            result['unsupported'].append({'path':path,'reason':'non_mesh_collider'})
            continue
        for mesh_prim in meshes:
            if approximation in {'sdf', 'meshSimplification'} and index is None:
                from .native_surface import native_surface_collider
                result['colliders'].append(native_surface_collider(env, stage, mesh_prim, path))
                continue
            if approximation == 'convexDecomposition' and index is None:
                from .native_convex import read_convex_parts, world_convex_colliders
                mesh_path = str(mesh_prim.GetPath())
                transform = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                result['colliders'].extend(world_convex_colliders(
                    read_convex_parts(mesh_path), mesh_path, path, transform))
                continue
            mesh = UsdGeom.Mesh(mesh_prim)
            points = array(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            triangles, offset = [], 0
            if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
                    or sum(counts) != len(indices) or np.any(counts<3)):
                result['unsupported'].append({'path':path,'reason':'invalid_mesh'})
                continue
            for count in counts:
                face=indices[offset:offset+count];offset+=count
                triangles.extend([[int(face[0]),int(face[i]),int(face[i+1])] for i in range(1,count-1)])
            try:
                points, triangles, representation = collision_envelope(points,triangles,approximation)
            except ValueError as error:
                from scipy.spatial import QhullError
                if index is not None or approximation != 'convexHull' or not isinstance(error.__cause__, QhullError):
                    raise
                # PhysX gives coplanar source meshes a cooked solid. Use that
                # exact native representation, never jitter or invent thickness.
                from .native_convex import read_convex_parts, world_convex_colliders
                mesh_path = str(mesh_prim.GetPath())
                transform = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                result['colliders'].extend(world_convex_colliders(
                    read_convex_parts(mesh_path), mesh_path, path, transform,
                    native_approximation='convexHull'))
                continue
            transform = np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            if index is not None:
                transform = np.linalg.inv(array(result['robots'][index]['body_frames'][body])) @ transform
            vertices = transform_vertices(points, transform, representation)
            result['colliders'].append({'id':str(mesh_prim.GetPath()),'collider_path':path,
                'robot_index':index,'body':body,'approximation':representation,
                'native_approximation':approximation,
                'geometry_source': 'usd_enclosing_convex_hull' if approximation == 'convexDecomposition' else 'usd_collision_geometry',
                'reference_world_transform':transform.tolist() if index is None else None,
                'vertices':vertices.tolist(),'triangles':triangles})
    return result


def forward_kinematics(robot, positions):
    positions = dict(zip(robot['joint_names'], positions))
    frames = {robot['base_link']: array(robot['body_frames'][robot['base_link']])}
    pending = list(robot['joints'])
    while pending:
        remaining = []
        for joint in pending:
            if joint['parent'] not in frames:
                remaining.append(joint)
                continue
            move=np.eye(4); axis=np.eye(3)['XYZ'.index(joint['axis'])]
            if joint['kind'] != 'fixed':
                value=positions[joint['name']]
                if joint['kind']=='prismatic':move[:3,3]=axis*value
                else:
                    x,y,z=axis;cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                    move[:3,:3]=np.eye(3)+np.sin(value)*cross+(1-np.cos(value))*(cross@cross)
            frames[joint['child']]=frames[joint['parent']]@array(joint['parent_frame'])@move@np.linalg.inv(array(joint['child_frame']))
        if len(remaining)==len(pending):
            raise ValueError('Incomplete physical joint tree')
        pending=remaining
    if set(frames)!=set(robot['body_frames']):
        raise ValueError('Physical body coverage is incomplete')
    return frames
