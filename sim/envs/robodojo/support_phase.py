"""Bootstrap native auxiliary actors without replacing native phase decisions.

Queue contents, task labels and phase counters remain operator-private. Later
queries/stability checks run exclusively inside the original take_action loop.
"""
from .robot_ownership import RobotOwnership


def initialize_support_phase(native):
    manager = getattr(native, 'robot_manager', None)
    if manager is None or not any(getattr(r, 'type', 'target') == 'support'
                                  for r in manager.robot_list):
        return False
    ownership = RobotOwnership.from_manager(manager)
    ownership.require_target_prefix()
    if len(ownership.support) != 1 or getattr(native, 'interact', None) is not True:
        raise ValueError('Unsupported native auxiliary actor phase contract')
    query = getattr(native, 'query_support_arm_traj', None)
    stable = getattr(native, 'check_support_arm_stable', None)
    running = getattr(native, 'get_running_env_idx_list', None)
    if not all(callable(fn) for fn in (query, stable, running)):
        raise ValueError('Native support phase callbacks are incomplete')
    queues = getattr(native, 'support_arm_action', None)
    if not isinstance(queues, list) or len(queues) != 1 or queues[0]:
        raise ValueError('Native support bootstrap requires one reset, empty environment queue')
    indices = list(running())
    if indices not in ([], [0]):
        raise ValueError('Support bootstrap only owns environment zero')
    for index in indices:
        query(env_idx=index)
    return True
