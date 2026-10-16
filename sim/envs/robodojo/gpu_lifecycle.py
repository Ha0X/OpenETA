"""Retire live articulations before deleting shapes invalidates GPU tensor views.

Preserves native per-object retirement operations and exclusion semantics.
No robot motion policy or active-episode collision coverage changes here.
"""
from functools import wraps


def install_scene_retirement_order(cls):
    original = cls.clear_scene_objects
    if getattr(original, '_openeta_gpu_retirement_order', False):
        return

    @wraps(original)
    def clear_scene_objects(self, env_id, exclude_types=None):
        excluded = exclude_types or []
        if getattr(self.device, 'type', str(self.device).split(':')[0]) == 'cuda' and 'articulation' not in excluded:
            # Native articulation retirement reads/writes its live state.
            # Native rigid deletion invalidates the shared PhysX tensor view.
            objects = self._articulation_objects[env_id]
            for key, obj in list(objects.items()):
                self.delete_scene_object(obj)
                del objects[key]
        return original(self, env_id, exclude_types=exclude_types)
    clear_scene_objects._openeta_gpu_retirement_order = True
    cls.clear_scene_objects = clear_scene_objects


def install_preallocated_layout_identity(cls):
    """Reuse identities of objects that native reset deliberately retains.

    CUDA retains rigid/dynamic objects and articulations; CPU retains only
    articulations. A different official layout requires a new environment when
    retained objects exist: native reset does not respawn or reconfigure them.
    CPU tasks without articulations keep native respawn/seed behavior.
    """
    import copy
    original_load = cls.load_saved_layout
    original_paths = cls._generate_object_paths
    if getattr(original_load, '_openeta_gpu_layout_identity', False):
        return

    @wraps(original_paths)
    def paths(self, env_idx, cat_name, model_idx, type):
        context = getattr(self, '_openeta_gpu_identity_context', None)
        if context is None or type not in context['retained_types']:
            return original_paths(self, env_idx, cat_name, model_idx, type)
        key = (env_idx, type, cat_name, model_idx)
        occurrence = context['counts'].get(key, 0)
        context['counts'][key] = occurrence + 1
        identity = (*key, occurrence)
        if context['replay']:
            if identity not in context['paths']:
                raise ValueError('Retained object identity changed; create a new environment')
            return context['paths'][identity]
        value = original_paths(self, env_idx, cat_name, model_idx, type)
        context['paths'][identity] = value
        return value

    @wraps(original_load)
    def load(self, env_idx):
        device = self.scene_manager.device
        layout = self.saved_layouts[env_idx]
        if layout is None:
            return original_load(self, env_idx)
        retained_types = ({'rigid', 'dynamic', 'articulation'}
                          if getattr(device, 'type', str(device).split(':')[0]) == 'cuda'
                          else {'articulation'})
        cache = getattr(self, '_openeta_gpu_identity_cache', {})
        previous = cache.get(env_idx)
        if previous is not None and previous['paths'] and layout != previous['layout']:
            raise ValueError('Worker retains objects from its initial official layout; create a new environment for a different layout')
        if previous is not None and not previous['paths']:
            previous = None
        context = {'paths': dict(previous['paths']) if previous else {},
                   'replay': previous is not None, 'counts': {}, 'retained_types': retained_types}
        if getattr(self, '_openeta_gpu_identity_context', None) is not None:
            raise ValueError('Concurrent layout reload is unsupported')
        self._openeta_gpu_identity_context = context
        try:
            result = original_load(self, env_idx)
            if previous is None:
                cache[env_idx] = {'layout':copy.deepcopy(layout), 'paths':dict(context['paths'])}
                self._openeta_gpu_identity_cache = cache
            return result
        finally:
            self._openeta_gpu_identity_context = None
    load._openeta_gpu_layout_identity = True
    cls._generate_object_paths = paths
    cls.load_saved_layout = load
