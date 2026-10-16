"""Preserve rewards evaluated inside native no-return take_action.

Based on Stage2 354b8cf9e73ecc027df31df2a8e67c1706d27b0e. The worker
serializes access to its owned environment. Never evaluate get_reward again:
native task predicates can change state when evaluated.
"""
from functools import wraps

import numpy as np


def finite_scalar(value, name):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in 'fiu' or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite numeric scalar')
    return float(array)


def _owned_value(values, name):
    array = np.asarray(values)
    if array.shape != (1,):
        raise ValueError(f'{name} must describe exactly one owned environment')
    return finite_scalar(array[0], name)


def step_with_native_reward(env, action):
    call = env.take_action if callable(getattr(env, 'take_action', None)) else env.step
    manager = getattr(env, 'reward_manager', None)
    get_reward = getattr(manager, 'get_reward', None)
    captured = {}
    if not callable(get_reward):
        return call(action), captured
    had_override = 'get_reward' in vars(manager)
    original = vars(manager).get('get_reward')

    @wraps(get_reward)
    def record(*args, **kwargs):
        rewards = get_reward(*args, **kwargs)
        captured['reward'] = _owned_value(rewards, 'native reward')
        captured.pop('score', None)
        scores = getattr(manager, '_gated_score_lst', None)
        if scores is not None:
            captured['score'] = _owned_value(scores, 'native score')
        return rewards

    manager.get_reward = record
    try:
        return call(action), captured
    finally:
        if had_override:
            manager.get_reward = original
        else:
            del manager.get_reward
