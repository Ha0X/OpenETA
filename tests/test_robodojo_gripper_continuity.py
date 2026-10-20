from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.gripper_continuity import GripperTargetContinuity


class Native:
    def __init__(self, signs=(1,)):
        robots=[];entities=[]
        for i,sign in enumerate(signs):
            robots.append(SimpleNamespace(ee_type='gripper',gripper_name=f'g{i}',gripper_scale=[-.01,.044],
                gripper_move={'sign':sign,'mimic':[None,-1.,.002]},gripper_joint_indices=[2,3]))
            target=.01 if sign==1 else .025
            entities.append(SimpleNamespace(root_physx_view=SimpleNamespace(
                get_dof_position_targets=lambda target=target:np.array([[0.,0.,target,-target+.002]]))))
        self.robot_manager=SimpleNamespace(robot_list=robots,robot_key=entities,process_name=lambda x:x)
        self.original_calls=0

    def process_control_info(self, info, env_idx):
        self.original_calls+=1
        sequence=[deepcopy(info) for _ in range(4)]
        for robot in self.robot_manager.robot_list:
            key=robot.gripper_name
            if key not in info:continue
            goal=info[key]['position'][0];start=.03 if robot.gripper_move['sign']==1 else .0
            for alpha,entry in zip([.1,.4,.7,1.],sequence):
                value=(1-alpha)*start+alpha*goal
                entry[key]['position']=[value,-value+.002]
        return sequence


def info(native, commands):
    result={'arm':{'position':[.4,.5], 'velocity':[.01,.02]}}
    for i,(r,g) in enumerate(zip(native.robot_manager.robot_list,commands)):
        goal=-.01+(g if r.gripper_move['sign']==1 else 1-g)*.054
        result[f'g{i}']={'position':[goal,-goal+.002],'velocity':0.}
    return result


@pytest.mark.parametrize('sign',[1,-1])
def test_same_closed_target_prevents_backoff_preserving_goal_mimic_and_other_fields(sign):
    native=Native((sign,));policy=GripperTargetContinuity(native,['primary']);command=info(native,[0.])
    with policy.action([0.]):first=native.process_control_info(command,0)
    assert not policy.feedback()
    with policy.action([0.]):second=native.process_control_info(command,0)
    assert native.original_calls==2
    prior=.01 if sign==1 else .025
    assert sign*(second[0]['g0']['position'][0]-prior)<=0
    assert second[-1]==first[-1]
    for entry in second:
        a,b=entry['g0']['position'];assert b==-a+.002
        assert entry['arm']==command['arm'] and entry['g0']['velocity']==0.
    event=policy.feedback()[0]
    assert event['role']=='primary' and event['adjusted_action_sequences']==1
    assert event['max_generated_target_backoff_prevented_m']>0
    assert not event['grasp_confirmed']
    assert 'process_control_info' not in vars(native)
    assert command==info(native,[0.])


def test_two_arms_only_repeated_closed_role_is_adjusted():
    native=Native((1,1));policy=GripperTargetContinuity(native,['left','right'])
    with policy.action([0.,0.]):pass
    command=info(native,[1.,0.]);expected=native.process_control_info(command,0)
    with policy.action([1.,0.]):result=native.process_control_info(command,0)
    assert [r['g0'] for r in result]==[r['g0'] for r in expected]
    assert result[0]['g1']['position'][0]==.01
    assert [e['role'] for e in policy.feedback()]==['right']


@pytest.mark.parametrize('commands',[[0.,.03],[0.,1.],[.5,.5],[0.,0.,1.,0.]])
def test_changed_partial_and_release_commands_use_original_ramp(commands):
    native=Native();policy=GripperTargetContinuity(native,['primary'])
    for index,value in enumerate(commands):
        command=info(native,[value]);expected=native.process_control_info(command,0)
        with policy.action([value]):result=native.process_control_info(command,0)
        if index==len(commands)-1:assert result==expected


def test_exception_restores_existing_override_and_does_not_commit_new_history():
    native=Native();original=native.process_control_info
    native.process_control_info=lambda info,index:original(info,index)
    override=native.process_control_info;policy=GripperTargetContinuity(native,['primary'])
    with policy.action([0.]):pass
    with pytest.raises(RuntimeError):
        with policy.action([0.]):
            native.process_control_info(info(native,[0.]),0)
            raise RuntimeError('dispatch failed')
    assert native.process_control_info is override
    assert policy.previous==[0.]
    with pytest.raises(RuntimeError):
        with policy.action([1.]):raise RuntimeError('new command failed')
    assert policy.previous==[0.]


def test_invalid_drive_target_rejected_without_sending_new_target():
    native=Native();policy=GripperTargetContinuity(native,['primary'])
    with policy.action([0.]):pass
    native.robot_manager.robot_key[0].root_physx_view.get_dof_position_targets=lambda:np.array([[0.,0.,np.nan,0.]])
    with pytest.raises(ValueError,match='drive target'):
        with policy.action([0.]):native.process_control_info(info(native,[0.]),0)
    assert 'process_control_info' not in vars(native)
    assert not policy.feedback()


def test_checked_action_applies_continuity_without_bypassing_physics_guard():
    from sim.envs.robodojo.checked_control import CheckedController
    class StepNative(Native):
        def sim_step(self):self.physics+=1
    native=StepNative();native.physics=0;c=CheckedController.__new__(CheckedController)
    c.native=native;c.roles=['primary'];c.recovery=None;c.joint_step_limited_actions=0
    c.world=SimpleNamespace(positions=lambda:[np.zeros(4)],scene=SimpleNamespace(snapshot={
        'robots':[{'joint_names':['a','b','g','m'],'arm_joint_names':['a','b']}]}))
    guarded=[];sequences=[]
    def guard(original):guarded.append(1);original()
    c._guarded_step=guard
    def step(action):
        sequences.append(native.process_control_info(info(native,[action['arms']['primary']['gripper']]),0))
        native.sim_step()
        return ('observed',)
    c.adapter=SimpleNamespace(step=step)
    for _ in range(2):c._action([np.array([.01,.02])],[0.])
    assert len(guarded)==native.physics==2
    assert sequences[0][0]['g0']['position'][0]>.01
    assert sequences[1][0]['g0']['position'][0]==.01
    assert c.last_step==('observed',)
    assert 'sim_step' not in vars(native) and 'process_control_info' not in vars(native)
    def fail(action):
        native.process_control_info(info(native,[0.]),0)
        raise RuntimeError('native action outcome unknown')
    c.adapter.step=fail
    with pytest.raises(RuntimeError):c._action([np.array([.01,.02])],[0.])
    assert 'sim_step' not in vars(native) and 'process_control_info' not in vars(native)
    assert len(guarded)==native.physics==2


def test_interleaved_task_owned_gripper_is_never_read_or_adjusted():
    native = Native((1, 1, 1))
    native.robot_manager.robot_list[1].type = 'support'
    def forbidden():
        raise AssertionError('Host must not read support drive targets for target continuity')
    native.robot_manager.robot_key[1].root_physx_view.get_dof_position_targets = forbidden
    policy = GripperTargetContinuity(native, ['left', 'right'])
    with policy.action([0., 0.]): pass
    command = info(native, [0., 0., 0.])
    expected = native.process_control_info(command, 0)
    with policy.action([0., 0.]):
        actual = native.process_control_info(command, 0)
    assert [entry['g1'] for entry in actual] == [entry['g1'] for entry in expected]
    assert {entry['role'] for entry in policy.feedback()} == {'left', 'right'}
