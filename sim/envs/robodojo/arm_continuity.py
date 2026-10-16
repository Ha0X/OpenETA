"""Continuous position setpoints for ordinary transport with closed hands.

The final target remains measured-state rate limited by CheckedController.
Every generated point lies on a segment checked by the existing physics guard.
Native drives, velocity limits, contact monitoring and episode time are intact.
"""
from contextlib import contextmanager

import numpy as np

from .scene_geometry import array


class ArmTargetContinuity:
    def __init__(self, control, grippers):
        self.control = control
        self.grippers = grippers

    @contextmanager
    def action(self):
        c = self.control
        if (getattr(c, 'operation', None) != 'move' or getattr(c, 'recovery', None)
                or getattr(c, 'support_placement', None) is not None):
            yield
            return
        selected = [i for i, (role, g) in enumerate(zip(c.roles, self.grippers))
                    if 0 <= g <= .05 and role in c.contact_grants
                    and role not in getattr(c, 'support_holds', {})]
        if not selected:
            yield
            return
        native = c.native
        manager = native.robot_manager
        previous = {}
        for i in selected:
            entity = manager.robot_key[i]
            schema = c.world.scene.snapshot['robots'][i]
            slots = [entity.joint_names.index(n) for n in schema['arm_joint_names']]
            value = array(entity.root_physx_view.get_dof_position_targets())[0, slots].copy()
            limits = array(schema['joint_limits'])[[schema['joint_names'].index(n)
                                                  for n in schema['arm_joint_names']]]
            if (value.shape != c.arm_targets[i].shape or not np.isfinite(value).all()
                    or np.any(value < limits[:, 0]-1e-6) or np.any(value > limits[:, 1]+1e-6)):
                raise ValueError('Previous arm drive target is not a valid calibrated joint target')
            previous[i] = value
        measured_starts = c.arm_start
        # Only the command segment's origin changes. _guarded_step still
        # certifies measured current joints -> next desired joints each tick.
        c.arm_start = [previous.get(i, q) for i, q in enumerate(measured_starts)]
        had_override = 'process_control_info' in vars(native)
        old_override = vars(native).get('process_control_info')
        original = native.process_control_info

        def process(info, env_idx):
            sequence = original(info, env_idx)
            if env_idx != 0 or not sequence:
                raise ValueError('Arm continuity requires one owned environment and a nonempty native interval')
            for i in selected:
                key = manager.process_name(manager.robot_list[i].arm_name)
                goal = array(info[key]['position'])
                # The adapter serializes joint-position actions as float32.
                # Compare that actual wire representation, not a tighter
                # arbitrary double-precision tolerance around it.
                if (goal.shape != c.arm_targets[i].shape or not np.array_equal(
                        goal.astype(np.float32), c.arm_targets[i].astype(np.float32))):
                    raise ValueError('Native arm goal differs from the checked motion target')
                changed = 0.
                for tick, entry in enumerate(sequence):
                    alpha = (tick+1)/len(sequence)
                    value = (1-alpha)*previous[i] + alpha*goal
                    changed = max(changed, float(abs(value-array(entry[key]['position'])).max()))
                    entry[key]['position'] = value.tolist()
                if changed > 1e-10:
                    if not hasattr(c, 'arm_continuity_events'):
                        c.arm_continuity_events = {}
                    role = c.roles[i]
                    event = c.arm_continuity_events.setdefault(role, {
                        'kind':'arm_target_continuity', 'role':role,
                        'adjusted_action_sequences':0, 'max_command_change_rad':0.,
                        'max_start_drive_offset_rad':0.,
                        'scope':'normal_motion_with_closed_authorized_hand; excludes_support_owner_and_contact_modes',
                        'interpolation':'previous_drive_target_to_same_checked_goal_over_full_native_interval',
                        'measured_target_increment_limits':'unchanged',
                        'native_drives_and_per_physics_step_checks':'unchanged',
                        'grasp_confirmed':False})
                    event['adjusted_action_sequences'] += 1
                    event['max_command_change_rad'] = max(event['max_command_change_rad'], changed)
                    event['max_start_drive_offset_rad'] = max(event['max_start_drive_offset_rad'],
                        float(abs(previous[i]-measured_starts[i]).max()))
            return sequence

        native.process_control_info = process
        try:
            yield
        finally:
            c.arm_start = measured_starts
            if had_override:
                native.process_control_info = old_override
            else:
                del native.process_control_info
