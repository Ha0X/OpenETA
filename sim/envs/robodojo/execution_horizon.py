"""Agent-selected simulation-time ceiling; never a speed or start-time command.

Whole native actions are rounded down. An independent physical-step guard
also bounds execution if an upstream action produces more steps than expected.
No scene/object state is used, and omission preserves the existing behavior.
"""
from decimal import Decimal, ROUND_FLOOR
import math


class ExecutionHorizon:
    def __init__(self, duration_s, *, physics_dt, steps_per_action):
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not 0 < duration_s <= 2.):
            raise ValueError('max_duration_s must be finite and in (0, 2]')
        if (isinstance(steps_per_action, bool) or not isinstance(steps_per_action, int)
                or not 1 <= steps_per_action <= 1000 or not math.isfinite(physics_dt) or physics_dt <= 0):
            raise ValueError('Execution horizon requires a verified native timebase')
        period = Decimal(str(physics_dt))*steps_per_action
        self.actions = int((Decimal(str(duration_s))/period).to_integral_value(rounding=ROUND_FLOOR))
        if self.actions < 1:
            raise ValueError(f'max_duration_s is shorter than one native action ({float(period):g}s)')
        self.requested = float(duration_s)
        self.dt = float(physics_dt)
        self.period = float(period)
        self.maximum_steps = self.actions*steps_per_action
        self.used_steps = 0

    def cap(self, existing=None):
        return min(existing, self.actions) if existing is not None else self.actions

    def exhausted(self):
        return self.used_steps >= self.maximum_steps

    def before_step(self, control):
        if not self.exhausted():
            return True
        control._halt('requested_execution_horizon_reached', preserve_gripper_targets=True)
        return False

    def after_step(self):
        self.used_steps += 1

    def decorate(self, control, receipt):
        # A new provider must not be allowed to silently bypass the physical guard.
        actual_steps = receipt.get('physics_steps', 0)
        if actual_steps != self.used_steps or self.used_steps > self.maximum_steps:
            raise RuntimeError('Execution horizon physical-step accounting mismatch')
        reached = receipt.get('reached_target') is True
        stop = receipt.get('stop')
        horizon_stop = self.exhausted() and not reached and (
            not stop or (stop.get('reason') == 'motion_budget_exhausted' and
                         stop.get('budget_kind') == 'native_actions'))
        if horizon_stop:
            # Hold measured arms and retain issued finger drive targets,
            # without a retreat, new closure command, or extra physics.
            # Keep the original budget evidence but distinguish requested return
            # from a collision/controller failure. No grasp claim is made.
            control.stopped = None
            control._halt('requested_execution_horizon_reached', preserve_gripper_targets=True)
            receipt['stop'] = dict(control.stopped)
            if stop:
                receipt['stop']['underlying_stop'] = stop
        receipt['execution_horizon'] = {
            'requested_max_duration_s': self.requested,
            'native_period_s': self.period,
            'scheduled_max_duration_s': self.maximum_steps*self.dt,
            'native_action_limit': self.actions,
            'actual_duration_s': self.used_steps*self.dt,
            'physics_steps': self.used_steps,
            'limit_reached': self.exhausted(),
            'timebase': 'simulation_physics_steps; excludes_planning_and_network_time',
            'target_changed': False, 'automatic_resume': False,
            'gripper_intent_preserved': True,
            'guidance': 'Inspect the returned measured state and images before another command. '
                        'A time-limited return does not imply target reached or grasp acquired. '
                        'If this ceiling stops the action, arms hold measured positions and existing finger '
                        'drive targets are retained when available (see stop.actuator_hold). '
                        'Gripper intent remains latched: later move/wait actions can continue it; '
                        'issue a gripper command to change that intent.'}
        receipt.setdefault('host_adjustments', []).append({
            'kind': 'agent_requested_execution_horizon',
            'quantization': 'down_to_whole_native_actions',
            'requested_max_duration_s': self.requested,
            'scheduled_max_duration_s': self.maximum_steps*self.dt})


def prepare(control, body):
    if 'max_duration_s' not in body:
        return None
    if (body.get('operation') not in ('move', 'gripper')
            or getattr(control, 'physical_contact', None) is None):
        raise ValueError('max_duration_s requires a physical-contact move or gripper request')
    count = control.native.obs_manager.collect_interval
    # Do not silently round a non-integral native configuration into a contract.
    import numbers
    if (not isinstance(count, numbers.Real) or isinstance(count, bool)
            or not math.isfinite(count) or not float(count).is_integer()):
        raise ValueError('Execution horizon requires an integral native action period')
    return ExecutionHorizon(body['max_duration_s'], physics_dt=float(control.native.sim.physics_dt),
                            steps_per_action=int(count))
