from types import SimpleNamespace as S
import pytest
from sim.envs.robodojo.support_phase import initialize_support_phase


def native():
    n = S(robot_manager=S(robot_list=[S(type='target'), S(type='target'), S(type='support')],
                           robot_key=[object()]*3), interact=True, support_arm_action=[[]])
    n.get_running_env_idx_list=lambda:[0]
    n.check_support_arm_stable=lambda **kw: (_ for _ in ()).throw(AssertionError('Not bootstrap'))
    n.query_support_arm_traj=lambda env_idx:n.support_arm_action[env_idx].append('native-frame')
    return n


def test_bootstrap_only_calls_native_query_once_and_preserves_contents():
    n=native(); assert initialize_support_phase(n)
    assert n.support_arm_action==[['native-frame']]
    with pytest.raises(ValueError,match='empty'):initialize_support_phase(n)


def test_target_only_and_ended_environment_do_not_start_support_actions():
    n=native(); n.robot_manager.robot_list=n.robot_manager.robot_list[:2]
    assert not initialize_support_phase(n)
    n=native(); n.get_running_env_idx_list=lambda:[]
    assert initialize_support_phase(n) and n.support_arm_action==[[]]


@pytest.mark.parametrize('change',['callback','foreign_environment','not_interactive'])
def test_incomplete_native_phase_contract_is_rejected(change):
    n=native()
    if change=='callback':n.check_support_arm_stable=None
    if change=='foreign_environment':n.get_running_env_idx_list=lambda:[0,1]
    if change=='not_interactive':n.interact=False
    with pytest.raises(ValueError):initialize_support_phase(n)
