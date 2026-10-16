"""Official terminal flags for the pinned RoboDojo EvalEnv contract.

EvalEnv.success starts True and is an execution-health flag while running.
Only end_flag=True after registered reward evaluation establishes an outcome.
This module never reads privileged object poses or changes native flags.
"""
from __future__ import annotations

import numpy as np

UPSTREAM_REVISION = 'ee67a1468510da7624a089164402359f2afc72c8'
SCHEMA = 'openeta.robodojo_official_outcome.v1'


def native_flag(env, name):
    value = getattr(env, name, None)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) != 1:
            raise ValueError(f'{name} must describe exactly one owned environment')
        value = value[0]
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must contain a boolean, not a truthy value')
    return bool(value)


def reward_registration_present(env):
    manager = getattr(env, 'reward_manager', None)
    # The native manager starts with empty lists and would award 1.0 on the
    # first action. Require actual task predicates, not just a callable method.
    # Final lists stay pending until final_check=True; query conditions are
    # also evaluated by get_reward. Preserve all native predicate categories.
    predicates = [getattr(manager, name, None) for name in (
        'check_list', 'final_check_list', 'trigger_check_list',
        'query_list', 'trigger_query_list')]
    return any(isinstance(value, list) and len(value) == 1 and
               isinstance(value[0], list) and bool(value[0]) for value in predicates)


def snapshot(env, *, reward_registered, action_index):
    end = native_flag(env, 'end_flag')
    health = native_flag(env, 'success')
    if not reward_registered:
        status = 'unknown'
    elif not end:
        status = 'running'
    else:
        status = 'success' if health else 'failure'
    return {'schema_version': SCHEMA, 'source': 'robodojo.eval_env.terminal_flags',
            'source_contract_revision': UPSTREAM_REVISION,
            'status': status, 'terminal': end, 'native_execution_ok': health,
            'reward_registered': bool(reward_registered), 'action_index': action_index,
            'benchmark_success': True if status == 'success' else False if status == 'failure' else None}
