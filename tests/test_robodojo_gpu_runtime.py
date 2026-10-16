from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.envs.robodojo.gpu_runtime import (
    configure, cpu_result, install_dof_properties, install_reward_pose_reader)


class DeviceTensor:
    """Reject implicit NumPy conversion like a CUDA tensor."""
    def __init__(self, value):self.value=np.asarray(value)
    def __array__(self, *a, **kw):raise TypeError('CUDA requires explicit host transfer')
    def detach(self):return self
    def cpu(self):return S(numpy=lambda:self.value)


def test_profile_is_opt_in_and_leaves_native_input_untouched():
    cfg={'sim':{'device':'cpu','use_fabric':False,'dt':.004,'physx':{'solver_type':1}},'reward':{'threshold':.1}}
    assert configure(cfg,'') is cfg
    changed=configure(cfg,'gpu_tensor_v1')
    assert changed['sim']['device']=='cuda:0' and changed['sim']['use_fabric']
    assert changed['reward']==cfg['reward'] and changed['sim']['physx']==cfg['sim']['physx']
    assert cfg['sim']['device']=='cpu' and not cfg['sim']['use_fabric']
    with pytest.raises(ValueError):configure(cfg,'cuda')


def test_dof_conversion_preserves_original_property_logic_and_does_not_patch_view():
    limits=DeviceTensor([[[-1,2],[-3,4]]]);gains=(DeviceTensor([[10,20]]),DeviceTensor([[1,2]]))
    view=S(get_dof_limits=lambda:limits,get_gains=lambda:gains)
    class Native:
        def __init__(self):self._articulation_view=view
        @property
        def dof_properties(self):
            lo=self._articulation_view.get_dof_limits()[0][:,0]
            stiff,damping=self._articulation_view.get_gains()
            return np.concatenate([lo,stiff[0],damping[0]])
    install_dof_properties(Native);getter=Native.dof_properties.fget;install_dof_properties(Native)
    assert Native.dof_properties.fget is getter
    assert np.array_equal(Native().dof_properties,[-1,-3,10,20,1,2])
    assert view.get_dof_limits() is limits


def test_reward_consumer_gets_same_values_without_changing_scene_reader():
    tensor=DeviceTensor([1.,2.,3.]);pose=(tensor,DeviceTensor([1.,0,0,0]))
    layout=S(get_instance_pose=lambda **kw:pose,marker=object())
    class Parser:
        def initialize(self, env):self.env=env;self.layout_manager=env.layout
        def native_reward(self):
            xyz,quat=self.layout_manager.get_instance_pose(inst_name='x')
            return float(xyz[2]>.1),np.concatenate([xyz,quat])
    install_reward_pose_reader(Parser);initialize=Parser.initialize;install_reward_pose_reader(Parser)
    assert Parser.initialize is initialize
    parser=Parser();env=S(layout=layout);parser.initialize(env)
    reward,observed=parser.native_reward();assert reward==1 and observed.tolist()==[1,2,3,1,0,0,0]
    assert env.layout is layout and env.layout.get_instance_pose()[0] is tensor
    assert parser.layout_manager.marker is layout.marker
    observed[0]=99;assert tensor.value[0]==1


def test_tuple_conversion_keeps_structure_and_dtype():
    result=cpu_result((DeviceTensor(np.array([2],dtype=np.float32)),np.array([3],dtype=np.int64)))
    assert isinstance(result,tuple) and result[0].dtype==np.float32 and result[1].dtype==np.int64
