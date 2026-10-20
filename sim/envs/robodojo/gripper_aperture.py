"""Robot-only nominal jaw clearance at the calibrated TCP section.

Reads only the two finger subtrees and their prismatic joints. This is an
asset-calibrated geometric measure, not object fitting or contact evidence.
"""
import numpy as np
from .scene_geometry import array, joint_frame

CONTRACT = 'openeta.x5_tcp_section_aperture.v1'


def section_bounds(vertices, triangles):
    """Projection of triangle/plane intersections onto the jaw axis (TCP X)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(triangles, dtype=int)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 3 or f.size == 0
            or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('Invalid finger triangle mesh')
    hits = []
    for tri in v[f]:
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if abs(a[2]) < 1e-9:
                hits.append(a[0])
            if a[2] * b[2] < 0:
                hits.append((a + (b-a)*(-a[2]/(b[2]-a[2])))[0])
    if not hits:
        raise ValueError('Finger does not cross the TCP section')
    return [float(min(hits)), float(max(hits))]


def calibrate(stage, root, tcp_from_link):
    from pxr import Usd, UsdGeom, UsdPhysics
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('Robot geometry must be in metres')
    tcp_inverse = np.linalg.inv(tcp_from_link)
    records = []
    for name, body, sign in [('joint7', 'link7', 1), ('joint8', 'link8', -1)]:
        jp = stage.GetPrimAtPath(root+'/joints/'+name)
        if not jp.IsA(UsdPhysics.PrismaticJoint):
            raise ValueError('Unverified prismatic finger joint')
        j = UsdPhysics.PrismaticJoint(jp)
        if ([str(p) for p in j.GetBody0Rel().GetTargets()] != [root+'/link6']
                or [str(p) for p in j.GetBody1Rel().GetTargets()] != [root+'/'+body]):
            raise ValueError('Unverified finger joint ownership')
        parent, child = joint_frame(j, 0), joint_frame(j, 1)
        axis = np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
        direction = tcp_inverse[:3,:3] @ parent[:3,:3] @ axis
        if not np.allclose(direction, [sign, 0, 0], atol=1e-6):
            raise ValueError('Finger does not slide along calibrated jaw axis')
        zero_to_tcp = tcp_inverse @ parent @ np.linalg.inv(child)
        bp = stage.GetPrimAtPath(root+'/'+body)
        body_world = np.array(UsdGeom.Xformable(bp).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        if not np.allclose(body_world[:3,:3].T @ body_world[:3,:3], np.eye(3), atol=1e-6):
            raise ValueError('Scaled robot asset is not calibrated')
        bounds = []
        for p in Usd.PrimRange(bp, Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Mesh) or '/collisions/' not in str(p.GetPath()):
                continue
            mesh = UsdGeom.Mesh(p)
            points = array(mesh.GetPointsAttr().Get())
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            if np.any(counts != 3) or len(indices) != sum(counts):
                raise ValueError('Expected triangulated X5 finger asset')
            world = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            local = np.linalg.inv(body_world) @ world
            if not np.allclose(local[:3,:3].T @ local[:3,:3], np.eye(3), atol=1e-6):
                raise ValueError('Scaled finger asset is not calibrated')
            t = zero_to_tcp @ local
            bounds.append(section_bounds(points @ t[:3,:3].T+t[:3,3], indices.reshape(-1,3)))
        if not bounds:
            raise ValueError('Missing finger collision mesh')
        limits = [float(j.GetLowerLimitAttr().Get()), float(j.GetUpperLimitAttr().Get())]
        if not np.allclose(limits, [0., .044], atol=1e-6):
            raise ValueError('Unverified finger travel limits')
        records.append({'joint':name, 'sign':sign, 'limits_m':limits,
                        'inner_at_zero_m': min(b[0] for b in bounds) if sign == 1 else max(b[1] for b in bounds)})
    gap = records[0]['inner_at_zero_m']-records[1]['inner_at_zero_m']
    if not 0 <= gap < .01:
        raise ValueError('Unexpected closed X5 section gap')
    return records


def measured_aperture(calibration, joint_positions):
    edges = []
    for r in calibration:
        q = float(joint_positions[r['joint']])
        low, high = r['limits_m']
        if not np.isfinite(q) or not low-1e-5 <= q <= high+1e-5:
            raise ValueError('Finger joint outside calibrated travel')
        edges.append(r['inner_at_zero_m']+r['sign']*q)
    closed = calibration[0]['inner_at_zero_m']-calibration[1]['inner_at_zero_m']
    return {'status':'available', 'calibration':CONTRACT,
            'width_m':edges[0]-edges[1],
            'width_limits_m':[closed, closed+sum(r['limits_m'][1] for r in calibration)],
            'inner_edges_tcp_x_m':[edges[1],edges[0]],
            'frame':'calibrated_tcp', 'section_tcp_z_m':0.,
            'basis':'robot_collision_mesh_section_and_both_measured_prismatic_joints',
            'scope':'nominal mesh opening at TCP plane, conservative projection across finger thickness',
            'limitation':'Other insertion depths may have different clearance; not object fit, collision margin, contact, force or grasp evidence.'}


def attach_aperture(env, entity, entry, tcp_from_link):
    """Cache robot geometry per owned environment; never scan the task scene."""
    try:
        import omni.usd
        root = str(entity.cfg.prim_path).replace('env_.*', 'env_0')
        cache = getattr(env, '_openeta_aperture_calibration', None)
        if cache is None:
            cache = env._openeta_aperture_calibration = {}
        if root not in cache:
            cache[root] = calibrate(omni.usd.get_context().get_stage(), root, tcp_from_link)
        q = array(entity.data.joint_pos)[0]
        values = {name: q[entity.joint_names.index(name)] for name in ('joint7','joint8')}
        report = measured_aperture(cache[root], values)
    except (ValueError, KeyError, IndexError, ImportError, AttributeError) as exc:
        report = {'status':'unavailable', 'reason':'robot_aperture_calibration_failed',
                  'detail':str(exc), 'width_m':None}
    entry.setdefault('gripper_state', {})['aperture'] = report
    entry['metadata']['gripper_aperture_calibration'] = CONTRACT if report['status']=='available' else 'unavailable'
