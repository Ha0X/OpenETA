from copy import deepcopy
import json

import pytest

from tools.harness_config import ConfigError, ConfigRegistry, ResolvedConfig, canonical
from tools.harness_state import named_robot_state, resolve_motion_scope
from test_harness_robodojo_adapter import raw_dual, Native, adapter


def dual():
    return ConfigRegistry().resolve('robodojo_dual_x5_v1')


def test_named_observation_preserves_roles_and_hides_object_privileges():
    raw = raw_dual()
    raw['objects'] = [{'name': 'SECRET_OBJECT', 'pose': [1,2,3]}]
    observed = adapter(Native())._convert_obs(raw, roles=('left','right'))
    result = named_robot_state(dual(), observed)
    assert set(result['effectors']) == {'left','right'}
    assert result['effectors']['right']['reported_pose']['xyz'] == [1,0,1]
    assert result['effectors']['left']['effector']['last_commanded_openness'] == .25
    assert 'openness' not in result['effectors']['left']['effector']
    assert not result['effectors']['left']['tcp_calibration_available']
    assert result['effectors']['left']['joints_by_name'] is None
    assert 'SECRET_OBJECT' not in json.dumps(result)


@pytest.mark.parametrize('root', [{'joint_positions': [0]*12}, {'arms': {'left': {}}}, {'arms': {'left': {}, 'right': {}, 'primary': {}}}])
def test_multi_arm_observation_cannot_be_flat_or_missing(root):
    with pytest.raises(ConfigError):
        named_robot_state(dual(), {'robot': root})


def test_joint_name_mapping_uses_names_instead_of_position():
    env = adapter(Native())
    obs = env._convert_obs(raw_dual(), roles=('left','right'))
    left = obs['proprio']['arms']['left']
    left.update(joint_positions=[1,2,3,4,5,6], joint_names=['j6','j5','j4','j3','j2','j1'])
    state = named_robot_state(dual(), obs)
    assert state['effectors']['left']['joints_by_name']['j6'] == 1


def graph(*, shared=False, hand=False):
    value = dual().value
    body = value['components']['embodiment']
    body['groups']['left_arm']['joint_names'] = ['left_joint', 'waist'] if shared else ['left_joint']
    body['groups']['right_arm']['joint_names'] = ['right_joint', 'waist'] if shared else ['right_joint']
    body['groups']['base'] = {'kind': 'mobile_base', 'joint_names': ['wheel_left','wheel_right']}
    for effector in body['effectors'].values():
        effector['affected_by_groups'].append('base')
    if hand:
        body['effectors']['left']['kind'] = 'dexterous_hand'
    return ResolvedConfig(canonical(value))


def test_independent_arm_scope_never_silently_adds_base_or_other_arm():
    result = resolve_motion_scope(graph(), target_effectors=['left'], participating_groups=['left_arm'])
    assert result['joint_write_set'] == ['left_joint']
    assert result['hold_effectors'] == []
    assert not result['scope_expanded'] and not result['execution_authorized']
    with pytest.raises(ConfigError) as error:
        resolve_motion_scope(graph(), target_effectors=['left'], participating_groups=['left_arm', 'base'])
    assert error.value.code == 'motion_scope_not_authorized'


def test_shared_waist_requires_holding_other_tcp_and_coupled_solver():
    result = resolve_motion_scope(graph(shared=True), target_effectors=['left'], participating_groups=['left_arm'])
    assert result['shared_joints'] == {'waist': ['left_arm','right_arm']}
    assert result['hold_effectors'] == ['right']
    assert result['requires_coupled_solver'] and not result['execution_authorized']


def test_unknown_joint_resources_do_not_mean_independent_arms():
    with pytest.raises(ConfigError) as error:
        resolve_motion_scope(dual(), target_effectors=['left'], participating_groups=['left_arm'])
    assert error.value.code == 'joint_resource_mapping_unknown'


def test_same_native_joint_names_in_distinct_articulations_are_not_shared_joints():
    config=ConfigRegistry().resolve('robodojo_dual_x5_checked_v1')
    result=resolve_motion_scope(config,target_effectors=['left'],participating_groups=['left_arm'])
    assert result['joint_write_set']==[f'left::joint{i}' for i in range(1,7)]
    assert result['shared_joints']=={}
    assert result['hold_effectors']==[] and not result['requires_coupled_solver']


def test_explicit_same_articulation_namespace_still_detects_shared_resource():
    value=graph(shared=True).value
    for group in value['components']['embodiment']['groups'].values():group['articulation']='humanoid'
    result=resolve_motion_scope(ResolvedConfig(canonical(value)),target_effectors=['left'],participating_groups=['left_arm'])
    assert result['shared_joints']=={'humanoid::waist':['left_arm','right_arm']}
    assert result['hold_effectors']==['right']


def test_explicit_mobile_base_participation_requires_other_hand_hold():
    value = graph().value
    value['components']['execution']['scope'].append('base')
    config = ResolvedConfig(canonical(value))
    result = resolve_motion_scope(config, target_effectors=['left'], participating_groups=['left_arm','base'])
    assert result['hold_effectors'] == ['right']
    assert result['requires_atomic_execution']
    assert not result['scope_expanded']


def test_hand_state_preserves_finger_joints_and_does_not_reuse_gripper_scalar():
    env = adapter(Native())
    obs = env._convert_obs(raw_dual(), roles=('left','right'))
    obs['proprio']['arms']['left']['finger_joint_positions'] = {'thumb': .2, 'index': .4}
    result = named_robot_state(graph(hand=True), obs)['effectors']['left']['effector']
    assert result == {'kind':'dexterous_hand','joint_positions':{'thumb':.2,'index':.4}}
