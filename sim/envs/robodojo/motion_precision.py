"""Explicit completion criterion; never changes requested pose or drive gains."""
import math
import numbers

DEFAULT_POSITION_TOLERANCE_M = .003
MIN_POSITION_TOLERANCE_M = .0005


def tolerance(value=DEFAULT_POSITION_TOLERANCE_M):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value)
            or not MIN_POSITION_TOLERANCE_M <= value <= DEFAULT_POSITION_TOLERANCE_M):
        raise ValueError('position_tolerance_m must be finite and in [0.0005, 0.003]')
    return float(value)


def decorate(receipt, control, targets, position_tolerance):
    from .checked_control import pose_error
    import numpy as np
    errors = {role:pose_error(control.tcp(control.roles.index(role)), np.asarray(path[-1]))
              for role,path in targets.items()}
    receipt['final_tcp_error'] = errors
    receipt['completion_criteria'] = {'position_tolerance_m':position_tolerance,
        'orientation_tolerance_rad':.03,'criterion_source':'request_or_profile_default',
        'target_modified':False,'drive_gains_modified':False,'task_success_inferred':False}
    within = all(e['position_m'] < position_tolerance and e['orientation_rad'] < .03
                 for e in errors.values())
    if receipt.get('reached_target') and not within:
        receipt['reached_target'] = False
        if receipt.get('status') in ('completed', 'completed_with_adjustment'):
            receipt['status'] = 'interrupted' if receipt.get('native_actions',0) else 'rejected'
        receipt.setdefault('host_adjustments',[]).append({'kind':'completion_rechecked_against_requested_precision',
            'reason':'measured_pose_outside_requested_tolerance'})
        if not receipt.get('stop'):
            receipt['stop'] = {'reason':'measured_final_pose_outside_requested_tolerance'}
    return receipt
