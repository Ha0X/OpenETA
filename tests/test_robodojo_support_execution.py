from copy import deepcopy
from types import SimpleNamespace as S

import pytest

from sim.envs.robodojo.robot_ownership import RobotOwnership
from sim.envs.robodojo.support_execution import SupportExecution


class Queue:
    def __init__(self): self.items = []
    def is_empty(self): return not self.items
    def pop(self): return self.items.pop(0)


class Manager:
    def __init__(self):
        self.control_queue = [Queue()]
        self.prev_control = [{'support_arm0': {'position': [-1]},
                              'support_ee0': {'position': [0.]}, 'left': 'target-before'}]
    def pop(self, env_idx_list=None):
        entry = self.control_queue[0].pop()
        self.prev_control[0].update(deepcopy(entry))
        return [entry]


class Native:
    def __init__(self):
        robots = [S(type='target', arm_name='left', gripper_name='left_ee'),
                  S(type='target', arm_name='right', gripper_name='right_ee'),
                  S(type='support', arm_name='support_arm0', gripper_name='support_ee0')]
        self.robot_manager = S(robot_list=robots, robot_key=[object()] * 3,
                               process_name=lambda x: x, control_manager=Manager())
        self.support_arm_action = [[{'support_arm0': {'position': [i]},
                                     'support_ee0': {'position': [i / 100.]}} for i in range(25)]]
        self.ownership = RobotOwnership.from_manager(self.robot_manager)
        self.physics_count = 0
    def process_control_info(self, info, env_idx):
        seq = [deepcopy(info) for _ in range(10)]
        for entry in seq:
            if self.support_arm_action[env_idx]:
                entry.update(self.support_arm_action[env_idx].pop(0))
        return seq


def ids(seq): return [entry['support_arm0']['position'][0] for entry in seq]


def execute(native, lease, limit):
    manager = native.robot_manager.control_manager
    with lease.action():
        seq = native.process_control_info({'left': 'target-after'}, 0)
        manager.control_queue[0].items.extend(seq)
        for i in range(len(seq)):
            manager.pop([0])  # Native pops/writes hold cache before sim_step.
            if i == limit:
                before = lease.pause()
                assert lease.pause() == before  # Repeated halt cannot duplicate.
                manager.control_queue[0].items.clear()
                break
            lease.begin_physics()
            native.physics_count += 1
            lease.commit_physics()
        # Native phase callbacks run here, before action context exit. Restored
        # frames must already be visible to their empty-queue checks.
        return ids(native.support_arm_action[0])


@pytest.mark.parametrize('completed', [0, 1, 2, 9, 10])
def test_pause_restores_exact_unexecuted_suffix_before_phase_callbacks(completed):
    n = Native(); lease = SupportExecution(n, n.ownership)
    before = deepcopy(n.support_arm_action[0])
    assert execute(n, lease, completed) == list(range(completed, 25))
    assert n.support_arm_action[0] == before[completed:]
    assert n.physics_count == completed
    m = n.robot_manager.control_manager
    assert m.prev_control[0]['support_arm0']['position'] == [completed-1]
    assert m.prev_control[0]['left'] == 'target-after'
    assert 'pop' not in vars(m) and 'process_control_info' not in vars(n)
    if completed < 10:
        assert lease.last_feedback['unexecuted_frames_restored'] == 10-completed
        assert lease.last_feedback['includes_popped_unexecuted_frame']
    assert execute(n, lease, 10) == list(range(completed+10, 25))


def test_after_physics_pause_does_not_replay_last_executed_frame():
    n = Native(); lease = SupportExecution(n, n.ownership); m = n.robot_manager.control_manager
    with lease.action():
        m.control_queue[0].items.extend(n.process_control_info({}, 0))
        m.pop([0]); lease.begin_physics(); lease.commit_physics()
        f = lease.pause(); m.control_queue[0].items.clear()
        assert not f['includes_popped_unexecuted_frame']
    assert ids(n.support_arm_action[0]) == list(range(1, 25))


def test_repeated_observation_windows_do_not_skip_duplicate_or_mutate_trajectory():
    n = Native(); lease = SupportExecution(n, n.ownership)
    for count in [2, 3, 1, 4, 5]:
        execute(n, lease, count)
    assert n.physics_count == 15
    assert ids(n.support_arm_action[0]) == list(range(15, 25))
    execute(n, lease, 10)
    assert n.support_arm_action == [[]] and n.physics_count == 25


def test_unknown_physics_never_restores_or_automatically_replays():
    n = Native(); lease = SupportExecution(n, n.ownership); m = n.robot_manager.control_manager
    with pytest.raises(RuntimeError, match='physical result'):
        with lease.action():
            m.control_queue[0].items.extend(n.process_control_info({}, 0))
            m.pop([0]); lease.begin_physics()
            raise RuntimeError('physical result uncertain')
    assert lease.invalid and lease.pause()['reset_required']
    assert ids(n.support_arm_action[0]) == list(range(10, 25))
    with pytest.raises(RuntimeError, match='reset'):
        with lease.action(): pass


def test_prephysics_exception_and_expansion_failure_restore_without_physics():
    n = Native(); lease = SupportExecution(n, n.ownership); m = n.robot_manager.control_manager
    with pytest.raises(ValueError):
        with lease.action():
            m.control_queue[0].items.extend(n.process_control_info({}, 0))
            m.pop([0]); raise ValueError('guard exception before physics')
    assert ids(n.support_arm_action[0]) == list(range(25)) and not lease.invalid
    original = n.process_control_info
    def broken(info, env):
        original(info, env)
        raise ValueError('native expansion interrupted')
    n.process_control_info = broken; m.control_queue[0].items.clear()
    with pytest.raises(ValueError, match='interrupted'):
        with lease.action(): n.process_control_info({}, 0)
    assert ids(n.support_arm_action[0]) == list(range(25))
    assert n.process_control_info is broken


def test_agent_cannot_supply_support_commands_or_pop_other_environment():
    n = Native(); lease = SupportExecution(n, n.ownership)
    with pytest.raises(ValueError, match='task-owned'):
        with lease.action(): n.process_control_info({'support_arm0': {'position': [99]}}, 0)
    assert ids(n.support_arm_action[0]) == list(range(25))
    with pytest.raises(ValueError, match='owned environment'):
        with lease.action(): n.robot_manager.control_manager.pop([1])


def test_role_partition_is_explicit_and_interleaved_profile_fails_closed():
    n = Native(); assert n.ownership.targets == (0, 1) and n.ownership.support == (2,)
    n.robot_manager.robot_list.reverse()
    owner = RobotOwnership.from_manager(n.robot_manager)
    assert owner.targets == (1, 2)
    with pytest.raises(ValueError, match='before support'):
        owner.require_target_prefix()
    n.robot_manager.robot_list[0].type = 'mystery'
    with pytest.raises(ValueError, match='Unknown'):
        RobotOwnership.from_manager(n.robot_manager)
