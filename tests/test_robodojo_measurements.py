from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.measurements import measure_effectors
from test_harness_robodojo_adapter import Native, adapter, action, raw_dual
from tools.harness_config import ConfigRegistry
from tools.harness_state import named_robot_state


def measured_native(sign=1):
    native = Native()
    robots = [SimpleNamespace(type='target',arm_name=f'{role}_arm',robot_name='x5',ee_type='gripper',
        gripper_joints_name=['joint7','joint8'],gripper_scale=[0.,.04],gripper_move={'sign':sign})
        for role in ('left','right')]
    calls = []
    def read(robot, env_idx_list):
        calls.append((robot.arm_name,env_idx_list))
        return {0: np.array([.01,.01]) if robot.arm_name=='left_arm' else np.array([.03,.03])}
    native.robot_manager = SimpleNamespace(robot_list=robots,get_end_effector_real_val=read)
    return native,calls


@pytest.mark.parametrize('sign,left,right', [(1,.25,.75),(-1,.75,.25)])
def test_native_joint_measurement_respects_model_sign_and_owned_index(sign,left,right):
    native,calls = measured_native(sign)
    measured = measure_effectors(native,('left','right'))
    assert measured['left']['openness'] == pytest.approx(left)
    assert measured['right']['openness'] == pytest.approx(right)
    assert measured['left']['joint_positions'] == {'joint7':.01,'joint8':.01}
    assert 'aperture_m' not in measured['left'] and 'grasped' not in measured['left']
    assert calls == [('left_arm',[0]),('right_arm',[0])]
    assert native.calls == []


def test_measured_position_does_not_replace_previous_command_for_other_hand():
    native,_ = measured_native()
    env = adapter(native)
    env.reset()
    # Actual right opening is now .25; the previous command remains .75.
    native.robot_manager.get_end_effector_real_val = lambda *a,**kw: {0: np.array([.01,.01])}
    observation,*_ = env.step(action())
    grip = observation['proprio']['arms']['right']['gripper_state']
    assert grip['openness'] == .25 and grip['commanded_openness'] == .75
    assert grip['position_error_from_command'] == -.5
    assert native.calls[0]['right_ee_joint_state'] == [.75]
    public = named_robot_state(ConfigRegistry().resolve('robodojo_dual_x5_v1'),observation)
    assert public['effectors']['right']['effector']['openness'] == .25
    assert public['effectors']['right']['effector']['last_commanded_openness'] == .75
    assert public['effectors']['right']['effector']['joint_positions'] == {'joint7':.01,'joint8':.01}
    assert public['effectors']['right']['effector']['measurement_source'] == 'native_joint_pos'


def test_outside_limits_is_reported_without_clamping_or_false_closure():
    native,_ = measured_native()
    native.robot_manager.get_end_effector_real_val = lambda *a,**kw: {0: [.05,.05]}
    grip = measure_effectors(native,('left','right'))['left']
    assert grip['measurement_available']
    assert 'openness' not in grip
    assert grip['unclipped_normalized_joint'] == 1.25


@pytest.mark.parametrize('measurement', [{1:[0,0]}, {0:[float('nan'),0]}, {0:[0]}])
def test_invalid_or_wrong_environment_measurements_remain_unknown(measurement):
    native,_ = measured_native()
    native.robot_manager.get_end_effector_real_val = lambda *a,**kw: measurement
    grip = measure_effectors(native,('left','right'))['left']
    assert not grip['measurement_available'] and 'openness' not in grip


def test_wrong_robot_cannot_supply_the_other_effectors_measurement():
    native,_ = measured_native()
    native.robot_manager.robot_list[1].arm_name = 'left_arm'
    assert all(not v['measurement_available'] for v in measure_effectors(native,('left','right')).values())


def test_dexterous_hand_does_not_get_scalar_gripper_normalization():
    native,calls = measured_native()
    native.robot_manager.robot_list[0].ee_type = 'hand'
    result = measure_effectors(native,('left','right'))
    assert all('openness' not in value for value in result.values())
    assert calls == []


def physical_native(sign=1, position=0.0):
    native,_ = measured_native(sign)
    for robot in native.robot_manager.robot_list:
        robot.gripper_scale = [-.01,.044]
    native.robot_manager.robot_key = [SimpleNamespace(joint_names=['joint8','joint7'],
        data=SimpleNamespace(joint_pos_limits=np.array([[[0.,.044],[0.,.044]]]))) for _ in range(2)]
    native.robot_manager.get_end_effector_real_val = lambda *a,**kw: {0:[position,position]}
    return native


@pytest.mark.parametrize('position,openness', [(0.,0.),(.022,.5),(.044,1.)])
def test_physical_stops_normalize_actual_opening_not_command_overtravel(position,openness):
    native=physical_native(position=position)
    result=measure_effectors(native,('left','right'))['left']
    assert result['openness'] == pytest.approx(openness)
    assert result['openness_basis'] == 'first_actuated_joint_physical_limits'
    assert result['normalization']['command_joint_limits'] == [-.01,.044]


def test_overtravel_is_reported_without_false_blocked_gripper():
    native=physical_native()
    env=adapter(native);env.reset()
    raw=raw_dual();raw['state']['left_ee_joint_state']=[0.0]
    obs=env._capture_observation(raw)
    grip=obs['proprio']['arms']['left']['gripper_state']
    assert grip['openness'] == 0
    assert grip['command_exceeds_physical_limits'] is True
    assert grip['commanded_joint_target'] == -.01
    assert grip['reachable_joint_target'] == 0
    assert grip['position_error_from_command'] == 0


def test_partial_and_reversed_commands_compare_in_same_physical_coordinates():
    native=physical_native(sign=-1,position=.017)
    env=adapter(native);env.reset()
    raw=raw_dual();raw['state']['left_ee_joint_state']=[.5]
    obs=env._capture_observation(raw)
    grip=obs['proprio']['arms']['left']['gripper_state']
    assert grip['openness'] == pytest.approx(1-.017/.044)
    assert grip['position_error_from_command'] == pytest.approx(0)
    assert grip['command_exceeds_physical_limits'] is False


def test_invalid_physical_mapping_preserves_raw_measurement_without_guessing():
    native=physical_native()
    native.robot_manager.robot_key[0].joint_names=['wrong','joint8']
    grip=measure_effectors(native,('left','right'))['left']
    assert grip['measurement_available'] is True
    assert 'openness' not in grip and 'physical_limits_unverified' in grip['normalization_reason']
