"""Private, hole-preserving covers of native PhysX scene-query mesh surfaces.

An anonymous, unattached kinematic owner lets the property-query cooker report
bounds for static colliders. No live stage attributes or simulation are changed.
The cover is a geometric collision proxy, NOT an exact SDF penetration oracle.
"""
import time
import math
import numpy as np

from .scene_geometry import array, pose_matrix
from .voxel_geometry import CoverUnavailable, surface_cover, cover_collider, native_query_domain


# Timed-out native queries may still own stage data. Release on their completion,
# never while native cooking is outstanding; a failed worker also releases it.
_PENDING_STAGES = {}


def await_query(finished, pump, clock, *, timeout_s=6., monotonic=time.monotonic):
    before = clock()
    deadline = monotonic() + timeout_s
    while not finished():
        if monotonic() >= deadline:
            raise CoverUnavailable('Native cooked bounds query timed out')
        pump()
        if clock() != before:
            raise CoverUnavailable('Read-only geometry query advanced physics')


def await_native_query(env, finished):
    """Process cooking callbacks using the verified IsaacLab render-only path."""
    import carb
    import omni.physx
    context = env.sim.sim
    if context.render.__module__ != 'isaaclab.sim.simulation_context':
        raise CoverUnavailable('Unverified render-only query pump')
    physics_events = []
    subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(
        lambda dt: physics_events.append(float(dt)))
    settings = carb.settings.get_settings()
    key = '/app/player/playSimulations'
    previous = settings.get(key)
    if not isinstance(previous, bool):
        raise CoverUnavailable('Unknown application physics-play setting')
    def clock():
        return (context.current_time_step_index, context.current_time, len(physics_events))
    try:
        settings.set_bool(key, False)
        await_query(finished, context.render, clock)
    finally:
        settings.set_bool(key, previous)
        subscription = None


def cooked_local_bounds(stage, mesh_prim, *, wait_for_completion=None):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils, Sdf, PhysicsSchemaTools
    import omni.physx
    from omni.physx.bindings._physx import PhysxPropertyQueryMode, PhysxPropertyQueryResult
    scratch = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(scratch, UsdGeom.GetStageMetersPerUnit(stage))
    UsdGeom.Xform.Define(scratch, '/Body')
    if not Sdf.CopySpec(stage.Flatten(), mesh_prim.GetPath(), scratch.GetRootLayer(), '/Body/Shape'):
        raise CoverUnavailable('Native query mesh copy failed')
    proxy = scratch.GetPrimAtPath('/Body/Shape')
    UsdGeom.Xformable(proxy).ClearXformOpOrder()
    body = UsdPhysics.RigidBodyAPI.Apply(scratch.GetPrimAtPath('/Body'))
    body.CreateKinematicEnabledAttr(True)
    cache = UsdUtils.StageCache.Get()
    stage_id = cache.Insert(scratch).ToLongInt()
    results, finished = [], []
    def receive(value):
        if value.result != PhysxPropertyQueryResult.VALID:
            results.append(None)
        else:
            # Copy all native callback storage before its lifetime ends.
            results.append((np.array(value.aabb_local_min),np.array(value.aabb_local_max),
                            np.array(value.local_pos),np.array(value.local_rot)))
    def complete():
        finished.append(True)
        pending = _PENDING_STAGES.pop(stage_id, None)
        if pending is not None:
            pending[0].Erase(pending[1])
    try:
        omni.physx.get_physx_property_query_interface().query_prim(
            stage_id=stage_id, prim_id=PhysicsSchemaTools.sdfPathToInt('/Body'),
            query_mode=PhysxPropertyQueryMode.QUERY_RIGID_BODY_WITH_COLLIDERS,
            collider_fn=receive, finished_fn=complete, timeout_ms=5000)
        if not finished and wait_for_completion is not None:
            wait_for_completion(lambda: bool(finished))
        if not finished or len(results)!=1 or results[0] is None:
            raise CoverUnavailable('Native cooked bounds query did not return one valid collider')
        low,high,position,xyzw=results[0]
        if (low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all()
                or np.any(high<=low) or not np.allclose(position,0,atol=1e-8)
                or not np.allclose(xyzw,[0,0,0,1],atol=1e-8)):
            raise CoverUnavailable('Unexpected native cooked bound frame')
        return np.array([low,high])
    finally:
        if finished:
            cache.Erase(scratch)
        else:
            _PENDING_STAGES[stage_id] = (cache, scratch)


def candidate_bound_ancestor(paths, bindings):
    """Candidate only: every native overlap hit MUST confirm this owner later.

    Native PhysX may ignore a nested USD RigidBodyAPI and assign its colliders
    to a parent body. Never treat an unbound hierarchy as a static collider.
    """
    if not paths:
        return None
    for path in paths:
        if path in bindings:
            return path
    raise CoverUnavailable('Native surface has no live rigid owner')


def query_delta(env, stage, prim):
    from pxr import Usd, UsdGeom, UsdPhysics
    from .physical_bindings import retain_rigid_bindings
    from .articulation_bindings import bind_articulations
    parent=prim
    paths=[]
    while parent.IsValid() and not parent.IsPseudoRoot():
        if parent.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(parent).GetRigidBodyEnabledAttr().Get():
            paths.append(str(parent.GetPath()))
        parent=parent.GetParent()
    roots={**retain_rigid_bindings(env),**bind_articulations(env)} if paths else {}
    path=candidate_bound_ancestor(paths,roots)
    if path is None:
        return np.eye(4),None
    parent=stage.GetPrimAtPath(path)
    reference=np.array(UsdGeom.Xformable(parent).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    scale=np.linalg.norm(reference[:3,:3],axis=0)
    if np.any(scale<=0):raise CoverUnavailable('Invalid native rigid scale')
    reference[:3,:3]/=scale
    if not np.allclose(reference[:3,:3].T@reference[:3,:3],np.eye(3),atol=2e-5):
        raise CoverUnavailable('Native surface rigid frame has shear')
    p,q=roots[path].get_world_pose()
    delta=pose_matrix([*array(p),*array(q)])@np.linalg.inv(reference)
    return delta,path


def cover_resolution(bounds, requested, *, static, max_cells=134217728):
    """Bounded conservative fixture proxy, never a contact-tolerance increase.

    Small and movable colliders retain the requested precision. Large STATIC
    fixtures may use aligned dyadic outer cells up to 4 mm; this can only
    over-report geometry. Narrow cavities may disappear, so expose the chosen
    precision rather than describing the proxy as exact native geometry.
    """
    bounds = np.asarray(bounds, dtype=float)
    if (bounds.shape != (2, 3) or not np.isfinite(bounds).all()
            or np.any(bounds[1] <= bounds[0]) or not math.isfinite(requested)
            or requested <= 0):
        raise CoverUnavailable('Invalid native cover precision request')
    resolution = requested
    while True:
        shape = np.ceil(bounds[1]/resolution)-np.floor(bounds[0]/resolution)+4
        if math.prod(map(int, shape)) <= max_cells:
            return resolution
        if not static or resolution*2 > .004:
            raise CoverUnavailable('Native cover cell budget exceeds bounded fixture precision')
        resolution *= 2


def native_surface_collider(env, stage, mesh_prim, collider_path, *, resolution=.0005):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    import omni.physx
    from scipy.spatial.transform import Rotation
    mode=mesh_prim.GetAttribute('physics:approximation').Get()
    if mode not in ('sdf','meshSimplification') or str(mesh_prim.GetPath())!=collider_path:
        raise CoverUnavailable('Native surface cover requires an explicitly owned mesh collider')
    bounds=cooked_local_bounds(stage,mesh_prim,wait_for_completion=lambda done:await_native_query(env,done))
    transform=np.array(UsdGeom.Xformable(mesh_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    if not np.isfinite(transform).all():raise CoverUnavailable('Invalid native mesh transform')
    margin=float(PhysxSchema.PhysxSDFMeshCollisionAPI(mesh_prim).GetSdfMarginAttr().Get()) if mode=='sdf' else 0.
    world_bounds=native_query_domain(bounds,UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(),transform,sdf_margin=margin)
    delta,owner=query_delta(env,stage,mesh_prim)
    if not np.allclose(delta[:3,:3].T@delta[:3,:3],np.eye(3),atol=2e-5):
        raise CoverUnavailable('Native surface query transform is not rigid')
    rotation=tuple(Rotation.from_matrix(delta[:3,:3]).as_quat())
    query=omni.physx.get_physx_scene_query_interface()
    path=str(mesh_prim.GetPath())
    static_owners=set()
    def intersects(center,half):
        matching=[]
        def collect(hit):
            if str(hit.collision)==path:
                if owner is not None and str(hit.rigid_body)!=owner:
                    matching.append('wrong_owner')
                elif owner is None and str(hit.rigid_body) not in static_owners:
                    reported=str(hit.rigid_body)
                    reported_prim=stage.GetPrimAtPath(reported) if reported else None
                    if reported and (not reported_prim.IsValid() or (
                            reported_prim.HasAPI(UsdPhysics.RigidBodyAPI) and
                            UsdPhysics.RigidBodyAPI(reported_prim).GetRigidBodyEnabledAttr().Get())):
                        matching.append('wrong_owner')
                    else:
                        static_owners.add(reported);matching.append(True)
                else:matching.append(True)
            return True
        current=delta[:3,:3]@center+delta[:3,3]
        query.overlap_box(tuple(half),tuple(current),rotation,collect,False)
        if 'wrong_owner' in matching:raise CoverUnavailable('Native surface query owner mismatch')
        return bool(matching)
    requested_resolution = resolution
    resolution = cover_resolution(world_bounds, resolution, static=owner is None)
    cover=surface_cover(world_bounds,intersects,resolution=resolution,max_cells=134217728,max_queries=8000000,timeout_s=180.)
    cover['requested_resolution_m'] = requested_resolution
    cover['conservative_static_fixture_fallback'] = resolution != requested_resolution
    cover['subresolution_openings_may_be_closed'] = True
    # The live world must remain fixed throughout the immutable cover build.
    final,final_owner=query_delta(env,stage,mesh_prim)
    if final_owner!=owner or not np.allclose(final,delta,atol=1e-10,rtol=0):
        raise CoverUnavailable('Native surface moved while its geometry was queried')
    item=cover_collider(cover,mesh_path=path,collider_path=collider_path,
                        reference_transform=transform,native_approximation=mode)
    item['native_verified_owner']=owner
    item['cover_evidence']['owner_verification']='all_matching_native_overlap_hits'
    item['cover_evidence']['bounds_source']='union_of_source_mesh_and_unattached_native_cooking_plus_sdf_domain_margin'
    item['cover_evidence']['oracle']='live_physx_overlap_box'
    item['cover_evidence']['sdf_penetration_equivalence']=False
    return item
