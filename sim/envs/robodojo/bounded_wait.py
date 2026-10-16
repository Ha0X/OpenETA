"""Bounded, billed settling through the native action/event path."""
import math
import time
import numpy as np


def wait(control, duration_s):
    if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or not 0 < duration_s <= 2.:
        raise ValueError('duration_s must be finite and in (0, 2]')
    if getattr(control, 'physical_contact', None) is None:
        raise ValueError('Wait currently requires physical_contact_v1')
    from .coordinated_motion import native_period
    from .worker_bridge import episode_budget
    dt, _, period = native_period(control)
    requested_actions = math.ceil(duration_s / period - 1e-10)
    control.operation = 'wait'
    control.stopped = None
    control.physics_steps = 0
    control.last_step = None
    control.recovery = None
    control.attachment_baseline = {}
    control.joint_step_limited_actions = 0
    control.world.refresh()
    positions = control.world.positions()
    targets = [q[[r['joint_names'].index(n) for n in r['arm_joint_names']]].copy()
               for q, r in zip(positions, control.world.scene.snapshot['robots'])]
    grippers = control._grippers()
    began = time.monotonic()
    count = 0
    reason = None
    while count < requested_actions:
        budget = episode_budget(control.adapter)
        if budget['terminal'] or (budget['remaining_actions'] is not None and budget['remaining_actions'] <= 0):
            reason = 'episode_terminal_or_budget_exhausted'
            break
        if time.monotonic() - began >= 90.:
            reason = 'wall_time_budget'
            break
        before = control.physics_steps
        control._action(targets[:len(control.roles)], grippers)
        count += 1
        if control.stopped:
            reason = 'physical_monitor_stop'
            break
        if control.physics_steps == before:
            reason = 'native_action_did_not_advance_physics'
            break
    actual = control.physics_steps * dt
    complete = actual + 1e-9 >= duration_s
    return {'schema_version': 'openeta.robodojo_wait.v1',
            'status': 'completed' if complete and not control.stopped else 'interrupted',
            'requested_duration_s': duration_s, 'actual_duration_s': actual,
            'native_period_s': period, 'requested_native_actions': requested_actions,
            'native_actions': count, 'physics_steps': control.physics_steps,
            'stop': control.stopped or ({'reason': reason} if reason else None),
            'hold_arm_joint_positions': dict(zip(control.roles, [q.tolist() for q in targets])),
            'gripper_intents': dict(zip(control.roles, grippers)),
            'host_adjustments': [{'kind': 'hold_measured_arm_configuration',
                'scope': 'measured_at_wait_start; latched_gripper_intents_preserved'},
                {'kind': 'duration_quantized_to_native_action_period',
                 'scheduled_duration_s': requested_actions * period}],
            'task_event_path': 'native_adapter_step_reward_and_counter',
            'force_control': False, 'task_success_inferred': False}
