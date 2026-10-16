"""Pinned runtime-only repair of a malformed native sphere rigid hierarchy.

The source USD, both collision shapes, masses, friction and reward code remain
owned by RoboDojo. Disable only duplicate child rigid-body *ownership* before
physics starts, so both shapes belong to the existing root rigid body.
"""
import hashlib
from functools import lru_cache, wraps
from pathlib import Path

SPHERE_SHA256 = 'c44c5e80df506501d59c66ad10dc9bb72ad8e7a6406d6b30468556eefc7ac0f9'


@lru_cache(maxsize=16)
def asset_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install():
    from env.scene_manager.objects.rigid import RigidObject
    from env.environment.isaac.isaac_rl_env import IsaacRLEnv
    from pxr import UsdGeom, UsdPhysics
    original = RigidObject._setup_physics
    if getattr(original, '_openeta_sphere_hierarchy', False):
        return
    initialize = IsaacRLEnv.__init__
    @wraps(initialize)
    def with_queries(self, cfg, *args, **kwargs):
        # Headless Isaac disables its scene-query manager by default. Enable
        # read-only ray/sweep queries before creating the physical scene.
        cfg.sim.enable_scene_query_support = True
        return initialize(self, cfg, *args, **kwargs)
    IsaacRLEnv.__init__ = with_queries

    @wraps(original)
    def repaired(self):
        if self.category_name == 'sphere':
            if asset_digest(self.usd_path) != SPHERE_SHA256:
                raise ValueError('Sphere asset revision changed; rigid hierarchy repair requires review')
            root = self.stage.GetPrimAtPath(self._prim_path)
            children = [self.stage.GetPrimAtPath(self._prim_path+'/'+name) for name in ('visual', 'collision')]
            for child in children:
                if (not child.IsValid() or not child.HasAPI(UsdPhysics.RigidBodyAPI)
                        or UsdGeom.Xformable(child).GetResetXformStack()):
                    raise ValueError('Pinned sphere rigid hierarchy has changed')
                mesh = self.stage.GetPrimAtPath(str(child.GetPath())+'/model')
                if not mesh.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get():
                    raise ValueError('Pinned sphere collision ownership has changed')
            if not root.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Sphere root rigid body is missing')
            for child in children:
                UsdPhysics.RigidBodyAPI(child).CreateRigidBodyEnabledAttr().Set(False)
            self._openeta_rigid_hierarchy_repair = 'pinned_sphere_children_owned_by_root_v1'
        return original(self)

    repaired._openeta_sphere_hierarchy = True
    RigidObject._setup_physics = repaired
