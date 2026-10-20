"""Honor native destruction for descendants, including referenced colliders.

This hook runs only when the native scene manager destroys an old object.
Active task objects keep their original collision flags and physics. Merely
hiding a parent or disabling CollisionAPI on that parent does not disable
independently authored child colliders.
"""
from functools import wraps


def disable_retired_colliders(stage, root_path):
    from pxr import Usd, UsdPhysics
    root=stage.GetPrimAtPath(root_path)
    if not root.IsValid():return []
    disabled=[]
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        attr=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        if attr.Get():
            if not attr.Set(False):
                raise ValueError('Unable to retire a native descendant collider')
            disabled.append(str(prim.GetPath()))
    return disabled


def install():
    from env.scene_manager.objects.rigid import RigidObject
    from env.scene_manager.objects.geometry import GeometryObject
    for cls in (RigidObject,GeometryObject):
        original=cls.destroy
        if getattr(original,'_openeta_retire_descendants',False):continue
        def wrap(method):
            @wraps(method)
            def destroy(self,*args,**kwargs):
                result=method(self,*args,**kwargs)
                self._openeta_retired_collision_paths=disable_retired_colliders(self.stage,self._prim_path)
                print('OPENETA_RETIRED_COLLIDERS '+str(len(self._openeta_retired_collision_paths)),flush=True)
                return result
            destroy._openeta_retire_descendants=True
            return destroy
        cls.destroy=wrap(original)
