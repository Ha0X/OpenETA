"""Keep an unchanged closing setpoint continuous across native action ramps.

Only robot command state is used. Native per-step limits and physical drives
remain downstream; this policy supplies no grasp or object-state evidence.
"""
from contextlib import contextmanager

import numpy as np

from .scene_geometry import array


class GripperTargetContinuity:
    def __init__(self, native, roles):
        self.native = native
        self.roles = list(roles)
        from .robot_ownership import RobotOwnership
        self.target_indices = RobotOwnership.from_manager(native.robot_manager).targets
        if len(self.target_indices) != len(self.roles):
            raise ValueError('Gripper roles must match only target robot owners')
        self.previous = [None] * len(roles)
        self.events = {}

    def feedback(self):
        return [dict(value) for value in self.events.values()]

    @contextmanager
    def action(self, commands):
        commands = list(commands)
        if len(commands) != len(self.roles):
            raise ValueError('Gripper continuity requires one command per effector')
        selected = [i for i, (old, new) in enumerate(zip(self.previous, commands))
                    if old is not None and 0 <= new <= .05 and abs(old-new) < 1e-9]
        if not selected:
            yield
            self.previous = commands
            return
        native = self.native
        original = native.process_control_info
        had_override = 'process_control_info' in vars(native)
        previous_override = vars(native).get('process_control_info')

        def process(info, env_idx):
            sequence = original(info, env_idx)
            manager = native.robot_manager
            for index in selected:
                robot = manager.robot_list[self.target_indices[index]]
                if robot.ee_type != 'gripper':
                    continue
                key = manager.process_name(robot.gripper_name)
                if key not in info:
                    continue
                low, high = map(float, robot.gripper_scale)
                sign = robot.gripper_move['sign']
                if sign not in (-1, 1) or not np.isfinite([low, high]).all() or high <= low:
                    raise ValueError('Unverified gripper closing direction or command range')
                goal = low + (commands[index] if sign == 1 else 1-commands[index])*(high-low)
                if not np.isclose(info[key]['position'][0], goal, atol=1e-8, rtol=0):
                    raise ValueError('Native gripper goal differs from the unchanged closing command')
                entity = manager.robot_key[self.target_indices[index]]
                indices = list(robot.gripper_joint_indices)
                if len(indices) != 2:
                    raise ValueError('Gripper continuity requires a calibrated parallel-jaw pair')
                previous = float(array(entity.root_physx_view.get_dof_position_targets())[env_idx, int(indices[0])])
                if not np.isfinite(previous) or not low-1e-6 <= previous <= high+1e-6:
                    raise ValueError('Previous gripper drive target is invalid')
                previous = float(np.clip(previous, low, high))
                backoff = 0.
                for entry in sequence:
                    before = float(entry[key]['position'][0])
                    value = max(goal, min(before, previous)) if sign == 1 else min(goal, max(before, previous))
                    backoff = max(backoff, sign*(before-value))
                    entry[key]['position'] = [value, value*robot.gripper_move['mimic'][1]+robot.gripper_move['mimic'][2]]
                if backoff > 1e-10:
                    role = self.roles[index]
                    event = self.events.setdefault(role, {
                        'kind': 'gripper_target_continuity', 'role': role,
                        'adjusted_action_sequences': 0,
                        'max_generated_target_backoff_prevented_m': 0.,
                        'scope': 'repeated_unchanged_full_closure_only',
                        'native_per_step_limiter_and_drives': 'unchanged',
                        'grasp_confirmed': False})
                    event['adjusted_action_sequences'] += 1
                    event['max_generated_target_backoff_prevented_m'] = max(
                        event['max_generated_target_backoff_prevented_m'], backoff)
            return sequence

        native.process_control_info = process
        try:
            yield
        finally:
            if had_override:
                native.process_control_info = previous_override
            else:
                del native.process_control_info
        self.previous = commands
