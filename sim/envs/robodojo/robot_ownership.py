"""Explicit command ownership; collision state retains every articulation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RobotOwnership:
    targets: tuple[int, ...]
    support: tuple[int, ...]

    @classmethod
    def from_manager(cls, manager):
        robots = manager.robot_list
        if len(robots) != len(manager.robot_key) or not robots:
            raise ValueError('Robot/entity ownership mapping is incomplete')
        kinds = [getattr(robot, 'type', 'target') for robot in robots]
        if any(kind not in ('target', 'support') for kind in kinds):
            raise ValueError('Unknown robot command owner')
        targets = tuple(i for i, kind in enumerate(kinds) if kind == 'target')
        support = tuple(i for i, kind in enumerate(kinds) if kind == 'support')
        if not targets:
            raise ValueError('No Agent-controllable target robot')
        return cls(targets, support)

    def require_target_prefix(self):
        # Existing planner indices are physical indices. Fail explicitly rather
        # than silently reindexing another native embodiment.
        if self.targets != tuple(range(len(self.targets))):
            raise ValueError('This checked profile requires target robots before support actors')

    def support_keys(self, manager):
        keys = tuple(manager.process_name(name) for i in self.support
                     for name in (manager.robot_list[i].arm_name, manager.robot_list[i].gripper_name))
        targets = {manager.process_name(name) for i in self.targets
                   for name in (manager.robot_list[i].arm_name, manager.robot_list[i].gripper_name)}
        if len(keys) != len(set(keys)) or targets.intersection(keys):
            raise ValueError('Native target/support command keys overlap')
        return keys
