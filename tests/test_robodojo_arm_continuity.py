import copy
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.arm_continuity import ArmTargetContinuity


def fixture():
    info={'left_arm':{'position':[.08,-.07]}, 'right_arm':{'position':[.2,.1]},
          'left_gripper':{'position':[.01,.01]}}
    def process(values, env_idx):
        result=[copy.deepcopy(values) for _ in range(3)]
        for i, fraction in enumerate([.5,1.,1.]):
            for name in ('left_arm','right_arm'):
                result[i][name]['position']=(np.array(values[name]['position'])*fraction).tolist()
        return result
    previous=np.array([[.02,-.01]])
    entities=[SimpleNamespace(joint_names=['a','b'],
        root_physx_view=SimpleNamespace(get_dof_position_targets=lambda:previous.copy())) for _ in range(2)]
    manager=SimpleNamespace(robot_key=entities,
        robot_list=[SimpleNamespace(arm_name=name) for name in ('left_arm','right_arm')],
        process_name=lambda name:name)
    native=SimpleNamespace(robot_manager=manager,process_control_info=process)
    schema={'joint_names':['a','b'],'arm_joint_names':['a','b'],'joint_limits':[[-2.,2.],[-2.,2.]]}
    c=SimpleNamespace(native=native,roles=['left','right'],operation='move',recovery=None,
        contact_grants={'left':{}},support_holds={},world=SimpleNamespace(scene=SimpleNamespace(
            snapshot={'robots':[schema,schema]})),arm_start=[np.zeros(2),np.zeros(2)],
        arm_targets=[np.array(info['left_arm']['position']),np.array(info['right_arm']['position'])])
    return c,info,previous


def test_closed_transport_starts_from_previous_drive_target_preserving_goal_and_guard_segment():
    c,info,previous=fixture();original=c.native.process_control_info
    initial_starts=c.arm_start;before=original(info,0)
    with ArmTargetContinuity(c,[0.,1.]).action():
        assert np.array_equal(c.arm_start[0],previous[0])
        generated=c.native.process_control_info(info,0)
        for index,entry in enumerate(generated):
            alpha=(index+1)/3
            assert np.allclose(entry['left_arm']['position'],
                c.arm_start[0]+alpha*(c.arm_targets[0]-c.arm_start[0]))
            assert entry['right_arm']==before[index]['right_arm']
            assert entry['left_gripper']==before[index]['left_gripper']
        assert generated[-1]['left_arm']==info['left_arm']
        assert c.arm_continuity_events['left']['adjusted_action_sequences']==1
        assert not c.arm_continuity_events['left']['grasp_confirmed']
    assert c.arm_start is initial_starts and c.native.process_control_info is original


@pytest.mark.parametrize('attribute,value,grippers', [
    ('operation','gripper',[0.,1.]), ('recovery',object(),[0.,1.]),
    ('support_placement',object(),[0.,1.]), ('support_holds',{'left':object()},[0.,1.]),
    ('contact_grants',{},[0.,1.]), ('operation','move',[1.,1.]),
    ('operation','move',[.3,1.]),
])
def test_contact_recovery_support_and_open_or_partial_commands_keep_native_interpolation(attribute,value,grippers):
    c,info,_=fixture();setattr(c,attribute,value);original=c.native.process_control_info
    with ArmTargetContinuity(c,grippers).action():
        assert c.native.process_control_info is original
    assert not getattr(c,'arm_continuity_events',{})


def test_exception_restores_command_anchor_and_wrapped_native_method():
    c,info,_=fixture();original=c.native.process_control_info;starts=c.arm_start
    with pytest.raises(RuntimeError,match='dispatch'):
        with ArmTargetContinuity(c,[0.,1.]).action():
            c.native.process_control_info(info,0)
            raise RuntimeError('dispatch')
    assert c.native.process_control_info is original and c.arm_start is starts


@pytest.mark.parametrize('invalid', [float('nan'),3.])
def test_invalid_previous_drive_state_fails_before_dispatch(invalid):
    c,_,previous=fixture();previous[0,0]=invalid
    with pytest.raises(ValueError,match='Previous arm drive target'):
        with ArmTargetContinuity(c,[0.,1.]).action():
            pytest.fail('must not dispatch')


def test_changed_native_goal_cannot_silently_replace_checked_target():
    c,info,_=fixture();info['left_arm']['position'][0]+=.01
    with ArmTargetContinuity(c,[0.,1.]).action():
        with pytest.raises(ValueError,match='differs from the checked'):
            c.native.process_control_info(info,0)


def test_adapter_float32_rounding_is_the_goal_identity_contract():
    c,info,_=fixture()
    c.arm_targets[0] = np.array([1.5123456789,-1.234567891])
    info['left_arm']['position'] = c.arm_targets[0].astype(np.float32).tolist()
    assert np.max(abs(np.array(info['left_arm']['position'])-c.arm_targets[0])) > 1e-8
    with ArmTargetContinuity(c,[0.,1.]).action():
        sequence = c.native.process_control_info(info,0)
        assert sequence[-1]['left_arm']['position'] == info['left_arm']['position']
