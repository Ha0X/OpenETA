"""Operator-selected episode protocol; immutable after first configured status."""
import numpy as np

UNLIMITED_ACTIONS = 'robodojo_unlimited_actions_v105'

EXTENDED_ALIGN = 'align_blocks_400_v1'
EXTENDED_TOWER = 'build_tower_2000_v1'
EXTENDED_TOWER_4000 = 'build_tower_4000_v1'
BUDGET_PROFILES = {
    UNLIMITED_ACTIONS: {'unlimited': True, 'limit': None, 'adapter_limit': None,
        'evaluation_variant': 'unlimited_action_budget'},
    'play_tic_tac_toe_1300_v104': {'task': 'play_tic_tac_toe',
        'native_original': 1100, 'limit': 1300, 'adapter_original': 1000, 'adapter_limit': 1300},
    'play_tic_tac_toe_native_adapter_v101': {'task': 'play_tic_tac_toe',
        'native_original': 1100, 'limit': 1100, 'adapter_original': 1000, 'adapter_limit': 1100,
        'evaluation_variant': 'native_limit_adapter_alignment'},
    'imitate_sorting_sequence_native_adapter_v101': {'task': 'imitate_sorting_sequence',
        'native_original': 1600, 'limit': 1600, 'adapter_original': 1000, 'adapter_limit': 1600,
        'evaluation_variant': 'native_limit_adapter_alignment'},
    EXTENDED_TOWER_4000: {'task': 'build_tower', 'native_original': 1050, 'limit': 4000,
                          'adapter_original': 1000, 'adapter_limit': 4000},
    EXTENDED_ALIGN: {'task': 'align_blocks', 'native_original': 200, 'limit': 400},
    EXTENDED_TOWER: {'task': 'build_tower', 'native_original': 1050, 'limit': 2000,
                     'adapter_original': 1000, 'adapter_limit': 2000},
}


def bind_protocol(adapter, requested, operation):
    current = getattr(adapter, '_episode_protocol', None)
    if current is not None:
        if current['override_id'] != requested:
            raise ValueError('Cannot change episode protocol within an episode')
        if requested and (adapter._env.step_lim != (float('inf') if current['effective_native_action_limit'] is None else current['effective_native_action_limit'])
                or adapter._max_episode_steps != (float('inf') if current['effective_adapter_action_limit'] is None else current['effective_adapter_action_limit'])):
            raise ValueError('Action limit drifted from bound episode protocol')
        return
    if requested is not None and requested not in BUDGET_PROFILES:
        raise ValueError('Unknown episode budget protocol')
    native = adapter._env
    original = getattr(native, 'step_lim', None)
    original = int(original) if isinstance(original, (int, np.integer)) and not isinstance(original, bool) and original > 0 else None
    adapter_original = adapter._max_episode_steps
    if type(adapter_original) is not int or adapter_original <= 0:
        raise ValueError('Invalid adapter quota')
    profile = BUDGET_PROFILES.get(requested)
    if profile:
        if (operation != 'status' or adapter._step_count != 0 or adapter._terminal
                or getattr(adapter, '_checked_controller', None) is not None):
            raise ValueError('Budget override requires initial status before any actions')
        if profile.get('unlimited'):
            if (original is None or adapter._robot_variant not in ('single_x5', 'dual_x5')
                    or np.shape(getattr(native, 'take_action_cnt', None)) != (1,)
                    or native.take_action_cnt[0] != 0):
                raise ValueError('Unlimited protocol requires a fresh native episode and known quotas')
        elif adapter._task_name != profile['task'] or adapter._robot_variant != 'dual_x5':
            raise ValueError('Budget override only supports dual X5 ' + profile['task'])
        counts = getattr(native, 'take_action_cnt', None)
        if not profile.get('unlimited') and (original != profile['native_original'] or np.shape(counts) != (1,) or counts[0] != 0
                or ('adapter_original' in profile and adapter_original != profile['adapter_original'])
                or ('adapter_limit' not in profile and adapter_original < profile['limit'])):
            raise ValueError('Unexpected native budget/counter or insufficient adapter quota')
        # Validate both layers first, then bind together. No reward/state mutation.
        native.step_lim = float('inf') if profile.get('unlimited') else profile['limit']
        adapter._max_episode_steps = float('inf') if profile.get('unlimited') else profile.get('adapter_limit', adapter_original)
    adapter._episode_protocol = {
        'override_id': requested,
        'evaluation_variant': profile.get('evaluation_variant', 'extended_action_budget') if profile else 'native_budget',
        'original_native_action_limit': original,
        'effective_native_action_limit': profile['limit'] if profile else original,
        'original_adapter_action_limit': adapter_original,
        'effective_adapter_action_limit': None if profile and profile.get('unlimited') else adapter._max_episode_steps,
        'success_predicates': 'unchanged_native',
        'comparable_to_native_budget_baseline': requested is None,
    }


def reset_protocol(adapter):
    previous = getattr(adapter, '_episode_protocol', None)
    if previous and previous['override_id']:
        adapter._env.step_lim = previous['original_native_action_limit']
        if 'original_adapter_action_limit' in previous:
            adapter._max_episode_steps = previous['original_adapter_action_limit']
    adapter._episode_protocol = None
