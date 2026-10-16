"""RoboDojo native flag regressions and atomically encoded dual actions.

The fixture models pinned EvalEnv semantics: success initially True while
end_flag False, take_action returns None, observations come from get_obs.
No Isaac runtime or synthetic task-success claims are required.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.unified_env import RoboDojoEnv
from sim.envs.robodojo.outcome import native_flag, snapshot


def raw_dual():
    return {'instruction': 'fixture task', 'vision': {}, 'state': {
        **{f'{role}_arm_joint_state': [i/10]*6 for i,role in enumerate(('left','right'))},
        'left_ee_pose': [0,0,1,1,0,0,0], 'right_ee_pose': [1,0,1,1,0,0,0],
        'left_ee_joint_state': [.25], 'right_ee_joint_state': [.75]}}


class Native:
    def __init__(self, status='running', register=True):
        self.status, self.register = status, register
        self.calls = []
        self.reset_calls = []
        self.registrations = 0
        self.closed = False

    def reset(self, seed):
        self.reset_calls.append(seed)
        self.success, self.end_flag = [True], [False]
        self.reward_manager = SimpleNamespace(check_list=[[]], final_check_list=[[]])
        self.reward_manager.get_reward = lambda: [1.0 if self.status == 'success' else 0.0]
        return raw_dual()

    def run_reward(self):
        self.registrations += 1
        if self.register:
            self.reward_manager.check_list[0].append('native task predicate')

    def get_obs(self):
        return raw_dual()

    def take_action(self, action):
        self.calls.append(deepcopy(action))
        self.success[0] = self.status != 'failure'
        self.end_flag[0] = self.status != 'running'
        self.reward_manager.get_reward()
        # Like pinned EvalEnv.take_action: no Gym tuple is returned.

    def close(self):
        self.closed = True


def adapter(native):
    return RoboDojoEnv('fixture', robot_variant='dual_x5', backend_env=native, interactive=True)


def action():
    return {'arms': {role: {'mode': 'joint_position', 'positions': [0]*6} for role in ('left','right')}}


def test_running_success_flag_is_not_benchmark_success():
    native = Native()
    env = adapter(native)
    _, reset = env.reset(seed=17)
    assert native.reset_calls == [[17]] and native.registrations == 1
    assert native.success == [True]
    assert reset['official_outcome']['status'] == 'running'
    obs,reward,ended,truncated,info = env.step(action())
    assert not ended and not truncated and reward == 0
    assert info['official_outcome']['benchmark_success'] is None
    assert 'success' not in info and 'benchmark_success' not in info


@pytest.mark.parametrize('status', ['success', 'failure'])
def test_real_native_termination_preserved_even_when_interactive(status):
    native = Native(status)
    env = adapter(native)
    env.reset()
    _,_,ended,_,info = env.step(action())
    assert ended and native.end_flag == [True]
    assert native.success == [status == 'success']
    assert info['benchmark_success'] is (status == 'success')
    assert info['official_outcome']['action_index'] == 1
    assert 'task_terminal_suppressed' not in info
    with pytest.raises(RuntimeError, match='ended'):
        env.step(action())
    assert len(native.calls) == 1


def test_empty_reward_manager_rejected_before_any_action():
    native = Native(register=False)
    env = adapter(native)
    with pytest.raises(RuntimeError, match='predicates are absent'):
        env.reset()
    with pytest.raises(RuntimeError, match='Reset'):
        env.step(action())
    assert native.calls == []


def test_final_only_predicate_is_not_mistaken_for_empty_manager():
    native = Native(register=False)
    native.run_reward = lambda: native.reward_manager.final_check_list[0].append('final predicate')
    env = adapter(native)
    assert env.reset()[1]['official_outcome']['status'] == 'running'


def test_dual_command_preserves_unselected_grippers_and_writes_once():
    native = Native()
    env = adapter(native)
    env.reset()
    request = action()
    request['arms']['left']['gripper'] = 0.0
    env.step(request)
    assert len(native.calls) == 1
    assert native.calls[0]['left_ee_joint_state'] == [0.0]
    assert native.calls[0]['right_ee_joint_state'] == [.75]
    assert set(native.calls[0]) == {'left_arm_joint_state','right_arm_joint_state','left_ee_joint_state','right_ee_joint_state'}


def test_no_gripper_guess_and_no_raw_bypass():
    env = adapter(Native())
    env.reset()
    del env._last_obs['proprio']['arms']['right']['gripper_command']
    with pytest.raises(ValueError, match='previous command'):
        env._to_native_action(action())
    with pytest.raises(ValueError, match='bypass'):
        env._to_native_action({'robodojo_action': {'anything': True}})


def test_public_xyzw_is_converted_once_to_native_wxyz():
    env = adapter(Native())
    env.reset()
    request = {'arms': {role: {'mode': 'ee_pose', 'pose': [0,0,1,.1,.2,.3,.4]} for role in ('left','right')}}
    before = deepcopy(request)
    encoded = env._to_native_action(request)
    expected = np.array([.4,.1,.2,.3])/np.linalg.norm([.1,.2,.3,.4])
    assert encoded['left_ee_pose'][3:] == pytest.approx(expected)
    assert request == before


def test_numpy_pose_is_not_mutated_while_encoding():
    env = adapter(Native())
    env.reset()
    pose = np.array([0,0,1,.1,.2,.3,.4], dtype=np.float32)
    before = pose.copy()
    env._to_native_action({'arms': {role: {'mode': 'ee_pose', 'pose': pose} for role in ('left','right')}})
    assert np.array_equal(pose, before)


@pytest.mark.parametrize('role', ['primary', 'left', 'right'])
def test_world_rotation_increment_is_applied_before_current_orientation(role):
    # R_x(90) @ R_z(90) differs from R_z(90) @ R_x(90); an identity
    # starting pose would miss this coordinate-frame regression.
    initial = np.array([0., 0., 1., 0., 0., 2**-.5, 2**-.5])
    _, encoded = RoboDojoEnv._arm_action(role,
        {'mode': 'ee_delta', 'delta_xyz': [.01, 0., 0.],
         'delta_rotvec': [np.pi/2, 0., 0.]}, {'ee_pose': initial})
    key = 'arm_ee_pose' if role == 'primary' else role+'_ee_pose'
    assert encoded[key][:3] == pytest.approx([.01, 0., 1.])
    # xyzw=[.5,-.5,.5,.5], converted once to native wxyz.
    assert encoded[key][3:] == pytest.approx([.5, .5, -.5, .5], abs=1e-6)
    assert initial.tolist() == [0., 0., 1., 0., 0., 2**-.5, 2**-.5]


def test_missing_gripper_observation_is_not_synthesized_closed():
    raw = raw_dual()
    del raw['state']['right_ee_joint_state']
    observed = adapter(Native())._convert_obs(raw, roles=('left','right'))
    assert 'gripper_open' not in observed['proprio']['arms']['right']
    assert 'gripper_command' not in observed['proprio']['arms']['right']


@pytest.mark.parametrize('gripper', [True, [0.1,0.2], -1, float('nan')])
def test_invalid_gripper_commands_rejected_before_native_action(gripper):
    native = Native()
    env = adapter(native)
    env.reset()
    request = action()
    request['arms']['left']['gripper'] = gripper
    with pytest.raises(ValueError):
        env.step(request)
    assert not native.calls


@pytest.mark.parametrize('bad', ['true', 1, [True,False], [None]])
def test_native_flag_never_uses_truthiness_or_other_environment(bad):
    with pytest.raises(ValueError):
        native_flag(SimpleNamespace(success=bad), 'success')


def test_step_budget_does_not_rewrite_native_outcome():
    native = Native()
    env = RoboDojoEnv('fixture', robot_variant='dual_x5', backend_env=native, max_episode_steps=1)
    env.reset()
    _,_,terminated,truncated,info = env.step(action())
    assert not terminated and truncated
    assert info['official_outcome']['status'] == 'running'
    assert info.get('benchmark_success') is None
    assert native.success == [True] and native.end_flag == [False]


def test_post_dispatch_observation_failure_blocks_blind_retry():
    native = Native()
    env = adapter(native)
    env.reset()
    native.get_obs = lambda: {}
    with pytest.raises(ValueError):
        env.step(action())
    with pytest.raises(RuntimeError, match='outcome is unknown'):
        env.step(action())
    assert len(native.calls) == 1
