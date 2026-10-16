"""Opt-in GPU tensor pipeline; CPU deployments never install these hooks.

Only conversion at NumPy consumers changes. Native assets, controller gains,
reward formulas, thresholds and action budgets remain native.
"""
import copy
import functools
import os
import numpy as np

PROFILE = 'gpu_tensor_v1'


def cpu_array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def cpu_result(value):
    if isinstance(value, tuple):
        return tuple(cpu_result(item) for item in value)
    return cpu_array(value)


class _NumpyView:
    def __init__(self, view):
        self._view = view

    def __getattr__(self, name):
        reader = getattr(self._view, name)
        if name not in {'get_dof_types', 'get_dof_limits', 'get_drive_types',
                        'get_max_efforts', 'get_joint_max_velocities', 'get_gains'}:
            raise AttributeError('Unreviewed DOF reader: '+name)
        return lambda *a, **kw: cpu_result(reader(*a, **kw))


class _PropertiesOwner:
    def __init__(self, owner):
        self._owner = owner
        self._articulation_view = _NumpyView(owner._articulation_view)

    def __getattr__(self, name):
        return getattr(self._owner, name)


def install_dof_properties(cls):
    original = cls.dof_properties
    if not isinstance(original, property):
        raise RuntimeError('Unsupported native dof_properties contract')
    if getattr(original.fget, '_openeta_gpu_numpy', False):
        return

    @functools.wraps(original.fget)
    def getter(self):
        return original.fget(_PropertiesOwner(self))
    getter._openeta_gpu_numpy = True
    cls.dof_properties = property(getter, original.fset, original.fdel, original.__doc__)


class _RewardPoseReader:
    """Only the NumPy reward consumer receives copied CPU poses."""
    def __init__(self, layout):
        self._layout = layout

    def get_instance_pose(self, *args, **kwargs):
        return cpu_result(self._layout.get_instance_pose(*args, **kwargs))

    def __getattr__(self, name):
        return getattr(self._layout, name)


def install_reward_pose_reader(cls):
    original = cls.initialize
    if getattr(original, '_openeta_gpu_numpy', False):
        return

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        self.layout_manager = _RewardPoseReader(self.layout_manager)
        return result
    initialize._openeta_gpu_numpy = True
    cls.initialize = initialize


def configure(env_cfg, profile=None):
    profile = os.environ.get('OPENETA_ROBODOJO_RUNTIME_PROFILE', '') if profile is None else profile
    if not profile:
        return env_cfg
    if profile != PROFILE:
        raise ValueError('Unknown RoboDojo runtime profile')
    value = copy.deepcopy(env_cfg)
    value['sim']['device'] = 'cuda:0'
    value['sim']['use_fabric'] = True
    return value


def install():
    profile = os.environ.get('OPENETA_ROBODOJO_RUNTIME_PROFILE', '')
    if not profile:
        return
    if profile != PROFILE:
        raise ValueError('Unknown RoboDojo runtime profile')
    from isaacsim.core.prims import SingleArticulation
    from env.reward_manager.func_parser import Func_Parser
    install_dof_properties(SingleArticulation)
    install_reward_pose_reader(Func_Parser)
    from env.scene_manager.scene_manager import SceneManager
    from .gpu_lifecycle import install_scene_retirement_order
    install_scene_retirement_order(SceneManager)
    from env.scene_manager.layout_manager import LayoutManager
    from .gpu_lifecycle import install_preallocated_layout_identity
    install_preallocated_layout_identity(LayoutManager)
    # Instrument the already installed endpoint's module-level operate lookup.
    from . import worker_bridge
    if getattr(worker_bridge.operate, '_openeta_gpu_runtime', False):
        return
    original = worker_bridge.operate

    @functools.wraps(original)
    def operate(adapter, body):
        if body.get('simulation_profile') != PROFILE:
            raise ValueError('GPU service requires explicit gpu_tensor_v1 profile')
        runtime = inspect_runtime(adapter._env)
        packet = original(adapter, body)
        packet['control']['simulation_runtime'] = runtime
        return packet
    operate._openeta_gpu_runtime = True
    worker_bridge.operate = operate


def inspect_runtime(native):
    from isaacsim.core.simulation_manager import SimulationManager
    view = SimulationManager.get_physics_sim_view()
    device = str(view.device)
    fabric = bool(native.use_fabric)
    if device != 'cuda:0' or str(native.device) != device or not fabric:
        raise ValueError('Requested GPU runtime differs from measured simulation')
    return {'profile':PROFILE, 'physics_tensor_device':device, 'use_fabric':fabric,
            'compatibility':'numpy_consumer_conversion_v1',
            'lifecycle':'articulations_retired_before_gpu_shape_deletion_v1',
            'sdf_collision_admission':False,
            'source':'live_physics_tensor_view_and_native_environment'}
