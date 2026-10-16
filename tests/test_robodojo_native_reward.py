from types import SimpleNamespace

import pytest

from sim.envs.robodojo.native_reward import step_with_native_reward
from test_harness_robodojo_adapter import Native, adapter, action


@pytest.mark.parametrize('reward', [0., .375, 1., -2.])
def test_no_return_native_reward_preserved_without_re_evaluation(reward):
    native = Native()
    env = adapter(native)
    env.reset()
    calls = []
    def evaluate():
        calls.append('evaluate')
        native.reward_manager._gated_score_lst = [.625]
        return [reward]
    native.reward_manager.get_reward = evaluate
    _, actual, terminated, truncated, info = env.step(action())
    assert actual == reward and calls == ['evaluate']
    assert native.reward_manager.get_reward is evaluate
    assert info['score'] == .625
    assert info['reward_source'] == 'robodojo_reward_manager'
    # Even reward=1 cannot replace the official terminal predicate.
    assert not terminated and not truncated
    assert info['official_outcome']['status'] == 'running'


def test_no_new_evaluation_cannot_reuse_previous_reward_or_retry_action():
    native = Native()
    env = adapter(native)
    env.reset()
    env.step(action())
    native.take_action = lambda request: native.calls.append(request)
    with pytest.raises(RuntimeError, match='no reward'):
        env.step(action())
    with pytest.raises(RuntimeError, match='outcome is unknown'):
        env.step(action())
    assert len(native.calls) == 2


@pytest.mark.parametrize('rewards', [[float('nan')], [True], [1,0], ['0.3']])
def test_invalid_native_reward_invalidates_post_dispatch_state(rewards):
    native = Native()
    env = adapter(native)
    env.reset()
    evaluator = lambda: rewards
    native.reward_manager.get_reward = evaluator
    with pytest.raises(ValueError):
        env.step(action())
    assert native.reward_manager.get_reward is evaluator
    with pytest.raises(RuntimeError, match='outcome is unknown'):
        env.step(action())
    assert len(native.calls) == 1


def test_class_method_restored_after_native_failure():
    class Manager:
        def get_reward(self):
            return [.5]
    manager = Manager()
    def fail(request):
        manager.get_reward()
        raise RuntimeError('native failed after dispatch')
    native = SimpleNamespace(reward_manager=manager, take_action=fail)
    with pytest.raises(RuntimeError, match='native failed'):
        step_with_native_reward(native, {})
    assert 'get_reward' not in vars(manager)
    assert manager.get_reward() == [.5]


def test_tuple_reward_keeps_native_tuple_semantics():
    native = Native()
    env = adapter(native)
    env.reset()
    def take(request):
        native.calls.append(request)
        return native.get_obs(), .25, False, False, {'success':True}
    native.take_action = take
    _, reward, _, _, info = env.step(action())
    assert reward == .25 and 'success' not in info
    assert info['reward_source'] == 'robodojo'
