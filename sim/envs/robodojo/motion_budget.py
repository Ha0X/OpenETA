"""Bounded execution allowances derived from the certified trajectory."""
import math

HARD_ACTION_LIMIT = 1200


def execution_budget(required, max_actions, timeout_s):
    required = int(required)
    tracking = max(32, math.ceil(required*.5))
    limit = min(HARD_ACTION_LIMIT, int(max_actions)) if max_actions is not None else min(
        HARD_ACTION_LIMIT, required+tracking)
    return {'required_nominal_actions': required, 'tracking_allowance_actions': max(0, limit-required),
            'action_limit': limit, 'hard_action_limit': HARD_ACTION_LIMIT,
            'wall_clock_limit_s': float(timeout_s), 'policy': 'trajectory_samples_plus_tracking_allowance',
            'explicit_action_limit': max_actions is not None}


def exhausted(budget, began, actions, now):
    kind = 'wall_clock' if now-began >= budget['wall_clock_limit_s'] else (
        'native_actions' if actions >= budget['action_limit'] else None)
    return ({'reason':'motion_budget_exhausted', 'budget_kind':kind,
             'elapsed_s':now-began, 'native_actions':actions, 'budget':budget} if kind else None)


def joint_action_estimate(trajectory, step_limit=.04):
    import numpy as np
    return sum(max(1, math.ceil(max(float(np.max(abs(np.asarray(b)-a))) for a,b in zip(start,end))/step_limit))
               for start,end in zip(trajectory,trajectory[1:]))
