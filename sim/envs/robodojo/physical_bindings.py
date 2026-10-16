"""Retain live handles for rigid bodies that survive native scene resets.

RoboDojo may replace its active object directory while previous USD/PhysX
bodies still exist. Those bodies remain collision obstacles, even offscreen.
Retaining their native read handles neither relocates nor disables them.
"""


def retain_rigid_bindings(native):
    bindings = dict(getattr(native, '_openeta_world_rigid_bindings', {}))
    manager = getattr(native, 'scene_manager', None)
    if manager is not None:
        groups = getattr(manager, '_rigid_and_dynamic_objects', [])
        if groups:
            for obj in groups[0].values():
                if not callable(getattr(obj, 'get_world_pose', None)):
                    raise ValueError('Native rigid object has no live pose reader')
                bindings[str(obj.prim_path)] = obj
                if type(obj).__module__ == 'env.scene_manager.objects.dynamic':
                    bindings.update(bind_dynamic_bodies(obj))
    native._openeta_world_rigid_bindings = bindings
    return bindings


class AssetRigidBody:
    """Exact native handle for a rigid body below an Xform-only asset wrapper."""
    def __init__(self, owner, view, path):
        self.owner, self.view, self.prim_path = owner, view, path
        if view.count != 1 or list(map(str, view.prim_paths)) != [path]:
            raise ValueError('Dynamic asset rigid view does not match its exact USD body')
        self.get_world_pose()

    def get_world_pose(self):
        import numpy as np
        from .scene_geometry import array
        if not self.owner.is_valid:
            raise ValueError('Dynamic asset physical handle is invalid')
        values = array(self.view.get_transforms())
        if values.shape != (1, 7) or not np.isfinite(values).all():
            raise ValueError('Dynamic asset rigid transforms are invalid')
        value = values[0]
        quat = value[[6,3,4,5]]
        if not np.isclose(np.linalg.norm(quat), 1., atol=1e-4):
            raise ValueError('Dynamic asset rigid quaternion is invalid')
        return value[:3].copy(), quat.copy()


def bind_dynamic_bodies(obj):
    """Read-only child-body binding; never use a wrapper Xform as live physics.

    Native DynamicObject deliberately preserves asset internals. Discover only
    enabled RigidBodyAPI descendants in that explicit asset, then require an
    exact one-body PhysX view. Global-frame tensor reads work even when USD
    transforms lag physics. Replaced assets get fresh handles; invalid retained
    handles fail closed. No collision or physics attributes are changed.
    """
    from pxr import Usd, UsdPhysics
    import omni.physics.tensors
    cached = getattr(obj, '_openeta_nested_rigid_bindings', None)
    if cached is not None:
        for reader in cached.values():
            reader.get_world_pose()
        return cached
    root = obj.stage.GetPrimAtPath(str(obj.prim_path))
    if not root.IsValid():
        raise ValueError('Dynamic asset root is missing')
    paths = [str(p.GetPath()) for p in Usd.PrimRange(root)
             if p.HasAPI(UsdPhysics.RigidBodyAPI)
             and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get()]
    bindings = {}
    if paths:
        owner = omni.physics.tensors.create_simulation_view('torch')
        owner.set_subspace_roots('/')
        for path in paths:
            bindings[path] = AssetRigidBody(owner, owner.create_rigid_body_view(path), path)
    obj._openeta_nested_rigid_bindings = bindings
    return bindings
