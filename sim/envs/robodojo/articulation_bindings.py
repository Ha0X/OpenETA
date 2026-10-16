"""Private live PhysX link bindings; no world joint state enters Agent sensors."""
import numpy as np
from .scene_geometry import array


class ArticulationLink:
    def __init__(self, owner, view, path, index, count):
        self.owner, self.view = owner, view
        self.prim_path, self.index, self.count = path, index, count

    def get_world_pose(self):
        if not self.owner.is_physics_handle_valid():
            raise ValueError('World articulation physical handle is invalid')
        transforms = array(self.view.get_link_transforms())
        if transforms.shape != (1, self.count, 7) or not np.isfinite(transforms).all():
            raise ValueError('World articulation link transforms are invalid')
        value = transforms[0,self.index]
        quat = value[[6,3,4,5]]  # PhysX xyzw -> existing collision adapter wxyz.
        if not np.isclose(np.linalg.norm(quat), 1., atol=1e-4):
            raise ValueError('World articulation quaternion is invalid')
        return value[:3].copy(), quat.copy()


def bind_articulations(native):
    bindings = dict(getattr(native, "_openeta_world_articulation_bindings", {}))
    seen = set()
    for obj in native.scene_manager._articulation_objects[0].values():
        owner = obj._articulation_view
        view = owner._physics_view
        paths = view.link_paths
        if len(paths) != 1 or len(paths[0]) != len(owner.body_names):
            raise ValueError('World articulation requires one verified link directory')
        paths = [str(p) for p in paths[0]]
        if len(set(paths)) != len(paths) or any(p.rsplit('/',1)[-1] != n for p,n in zip(paths,owner.body_names)):
            raise ValueError('World articulation link names do not match physical paths')
        # SingleArticulation.prim_path can resolve to the root BODY (e.g.
        # laptop/base), whose sibling screen is a legal link. The native asset
        # loader retains its authored instance root in usd_prim_path/_prim_path.
        root = str(getattr(obj, 'usd_prim_path', obj.prim_path)).rstrip('/')
        physical_root = str(obj.prim_path).rstrip('/')
        if (not root or not root.startswith('/')
                or (hasattr(obj, 'usd_prim_path') and str(obj._prim_path).rstrip('/') != root)
                or not (physical_root == root or physical_root.startswith(root+'/'))):
            raise ValueError('Articulation physical root is outside its authored scene object')
        for index, path in enumerate(paths):
            if path != root and not path.startswith(root+'/'):
                raise ValueError('Articulation physical link is outside its scene object')
            if path in seen:
                raise ValueError('Duplicate physical articulation link')
            seen.add(path)
            link = ArticulationLink(owner,view,path,index,len(paths))
            link.get_world_pose()
            bindings[path] = link
    native._openeta_world_articulation_bindings = bindings
    return bindings
