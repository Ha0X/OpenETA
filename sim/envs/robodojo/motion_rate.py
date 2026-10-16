"""Command increments from public robot intent; native drive limits stay intact."""
import math

TOOL_CONTACT_JOINT_STEP_LIMIT_RAD = .01


def joint_step_limits(control, grippers):
    if len(grippers) != len(control.roles) or not all(math.isfinite(g) and 0 <= g <= 1 for g in grippers):
        raise ValueError('Motion rate requires one finite gripper command per role')
    placement = getattr(control, 'support_placement', None)
    bound = (TOOL_CONTACT_JOINT_STEP_LIMIT_RAD if hasattr(placement, 'workpieces') else .01 if placement is not None
             else .02 if getattr(control, 'recovery', None) else .04)
    if bound != .04 or getattr(control, 'operation', None) != 'move':
        return [bound] * len(control.roles)
    # Closed/partly closed or contact-authorized hands keep the held-motion
    # bound, even if the attachment detector currently has no evidence.
    grants = getattr(control, 'contact_grants', {})
    return [.08 if math.isfinite(g) and .99 <= g <= 1. and role not in grants else .04
            for role, g in zip(control.roles, grippers)]


def record_rate(control, limits, starts, targets):
    events = getattr(control, 'motion_rate_events', None)
    if events is None:
        events = control.motion_rate_events = {}
    for role, limit, start, target in zip(control.roles, limits, starts, targets):
        if limit <= .04 or max(abs(target-start)) <= .008:
            continue
        event = events.setdefault(role, {
            'kind': 'open_gripper_motion_rate', 'role': role,
            'joint_step_limit_rad': .08, 'previous_joint_step_limit_rad': .04,
            'condition': 'fully_open_command_and_no_grasp_contact_authorization',
            'applied_native_actions': 0,
            'native_drives_and_collision_checks': 'unchanged',
            'empty_hand_confirmed': False,
            'action_budget_estimate': 'conservative_0.04_rad_bound_unchanged'})
        event['applied_native_actions'] += 1
