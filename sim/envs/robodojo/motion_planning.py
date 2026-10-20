"""Preflight whole waypoint requests, then track certified joint segments."""
import itertools
import time
from collections import deque

import numpy as np

from .scene_geometry import array
from .motion_budget import execution_budget, exhausted, joint_action_estimate

CARTESIAN_EDGE_CHECK_TIMEOUT_S = 4.


def combined_goals(control, poses, start, scene, selected, deadline):
    choices, diagnostics = [], []
    for i, schema in enumerate(scene.snapshot['robots']):
        slots = [schema['joint_names'].index(n) for n in schema['arm_joint_names']]
        if i not in selected:
            choices.append([array(start[i])[slots]])
            continue
        if time.monotonic() >= deadline:
            return [], {'reason': 'ik_preview_budget', 'status': 'unknown'}
        candidates, detail = control._pose_candidates(i, poses[i], start[i])
        diagnostics.append(detail)
        if not candidates:
            return [], {'reason': 'ik_search_no_valid_solution', 'status': 'unknown',
                        'role': control.roles[i], 'ik': detail}
        choices.append(candidates)
    goals, rejected, first_rejected = [], 0, None
    tested=0; exhausted=False
    combinations = sorted(itertools.product(*choices), key=lambda combo: sum(
        np.linalg.norm(q-array(start[i])[[scene.snapshot['robots'][i]['joint_names'].index(n)
                          for n in scene.snapshot['robots'][i]['arm_joint_names']]]) for i, q in enumerate(combo)))
    for combo in combinations[:64]:
        if time.monotonic() >= deadline:
            exhausted=True
            break
        tested+=1
        full = [array(q).copy() for q in start]
        for i, q in enumerate(combo):
            schema = scene.snapshot['robots'][i]
            full[i][[schema['joint_names'].index(n) for n in schema['arm_joint_names']]] = q
        check = scene.check(full, allowed_pairs=control.allowed_pairs, clearance_limit=.001)
        if check['clear']:
            goals.append(full)
        else:
            rejected += 1
            if first_rejected is None:
                first_rejected = (tested, check)
    return goals, {'reason': 'ik_candidates_checked' if goals else 'ik_preview_budget' if exhausted else 'ik_candidates_in_collision',
        'status': 'clear' if goals else 'unknown', 'ik': diagnostics,
        'collision_rejected_candidates': rejected, 'admissible_candidates': len(goals),
        'candidate_combinations_generated':len(combinations),'candidate_combinations_tested':tested,
        'candidate_combination_limit':64,'candidate_search_truncated':tested<len(combinations),
        'detail': first_rejected[1] if not goals and first_rejected else None,
        'collision_evidence': ({'selection':'least_joint_displacement_among_tested_rejected_candidates',
            'candidate_rank':first_rejected[0], 'scope':'one_rejected_ik_configuration_not_all_candidates'}
            if not goals and first_rejected else None),
        'guidance': 'Finite candidate search is not proof that this pose is unreachable.'}


def plan_request(control, targets, timeout_s):
    from .checked_control import cartesian_knots
    from .attached_geometry import attached_scene
    started = time.monotonic(); deadline = started+timeout_s
    control.world.refresh()
    start = control.world.positions()
    scene, attachments = attached_scene(control.world, start, control.contact_grants)
    scene=control._contact_scene(scene)
    # Dense robot meshes can require several seconds even for a clear edge.
    # This remains capped by the shared request deadline and node budget.
    edge_check_limit = CARTESIAN_EDGE_CHECK_TIMEOUT_S
    selected = [control.roles.index(role) for role in targets]
    schema = scene.snapshot['robots']
    poses = [control.tcp(i) for i in range(len(control.roles))]
    initial_check = scene.check(start, allowed_pairs=control.allowed_pairs, clearance_limit=.001)
    if not initial_check['clear']:
        # Contact recovery remains a separate, explicitly short request.
        return None, {'reason': 'initial_collision_requires_recovery', 'status': 'blocked',
                      'detail': initial_check, 'attachments': attachments}
    if initial_check['clearance_m'] <= .0002:
        from .clearance_recovery import near_clearance_pairs
        near, detail = near_clearance_pairs(scene,start,control.allowed_pairs)
        if near:
            return None, {'reason':'initial_clearance_requires_recovery','status':'blocked',
                          'detail':detail,'attachments':attachments}
    trajectory, stages, previews = [start], [], []
    for stage in range(max(map(len, targets.values()))):
        goals = [array(targets[role][min(stage, len(targets[role])-1)]) if role in targets else poses[i]
                 for i, role in enumerate(control.roles)]
        endpoint_candidates, detail = combined_goals(control, goals, start, scene, selected, deadline)
        previews.append(detail)
        if not endpoint_candidates:
            if detail.get('reason')=='ik_candidates_in_collision':
                detail={**detail,'guidance':'All checked IK candidates at this endpoint collided. A different route alone does not change those endpoint configurations. Consider a different target position/orientation or repositioning a blocking arm before rechecking. Finite search does not prove infeasibility.'}
            return None, {**detail, 'phase': 'endpoint_preview', 'endpoint': stage, 'physics_advanced': False}
        knots = [cartesian_knots(a, b) for a, b in zip(poses, goals)]
        local, cursor, failed = [], start, None
        cartesian_deadline = deadline
        for k in range(max(map(len, knots))):
            if time.monotonic() >= cartesian_deadline:
                failed={'status':'unknown','reason':'cartesian_preview_budget_exhausted'}
                break
            failed = None
            knot = [p[min(k, len(p)-1)] for p in knots]
            candidates, detail = combined_goals(control, knot, cursor, scene, selected, cartesian_deadline)
            chosen = None
            for candidate in candidates[:8]:
                result = scene.check_path(cursor, candidate, allowed_pairs=control.allowed_pairs,
                    timeout_s=min(edge_check_limit, max(0., cartesian_deadline-time.monotonic())), max_nodes=512)
                if result['status'] == 'clear':
                    chosen = candidate
                    break
                failed = result
            if chosen is None:
                failed = failed or detail
                break
            local.append(chosen); cursor = chosen
        else:
            trajectory.extend(local); start = cursor; poses = goals
            stages.append({'endpoint': stage, 'mode': 'cartesian_knots', 'joint_knots': len(local)})
            continue
        return None, {'reason': 'requested_cartesian_path_not_certified',
            'status': 'unknown' if (failed or {}).get('status') == 'unknown' else 'blocked',
            'phase': 'path_preview', 'endpoint': stage, 'knot_index': k,
            'physics_advanced': False, 'cartesian_check': failed,
            'cartesian_edge_check_time_limit_s':edge_check_limit,
            'automatic_spatial_replanning': False,
            'guidance': 'The requested path was not certified. No alternative route was executed. Inspect the reported collision or IK evidence and choose the next target/waypoints. A finite search failure does not prove impossibility.'}
    return trajectory, {'status': 'clear', 'stages': stages, 'endpoint_previews': previews,
        'attachments': attachments, 'elapsed_s': time.monotonic()-started,
        'cartesian_edge_check_time_limit_s':edge_check_limit,
        'planner': 'cartesian_knots_with_checked_ik',
        'automatic_spatial_replanning': False,
        'collision_guarantee': 'stationary_world_swept_joint_segments; conditional_rigid_attachment'}


def validate_motion_targets(control, targets):
    if not targets or not set(targets) <= set(control.roles):
        raise ValueError('Unknown or missing effector roles')
    for path in targets.values():
        if not 1 <= len(path) <= 5:
            raise ValueError('Supply one destination and at most four waypoints')
        for p in path:
            p = array(p)
            if (p.shape != (4, 4) or not np.isfinite(p).all() or not np.allclose(p[3], [0, 0, 0, 1])
                or not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(p[:3, :3]), 1, atol=1e-6)):
                raise ValueError('TCP target must be a finite rigid transform')


def execute_request(control, targets, *, timeout_s, max_actions):
    from .checked_control import pose_error
    from .motion_rate import TOOL_CONTACT_JOINT_STEP_LIMIT_RAD
    validate_motion_targets(control, targets)
    from .motion_precision import tolerance
    position_tolerance = tolerance(getattr(control, 'position_tolerance_m', .003))
    control.stopped = None; control.physics_steps = 0; control.last_step = None
    control.operation = 'move'; control.attachment_baseline = {}
    control.recovery = None; control.joint_step_limited_actions = 0
    started = time.monotonic()
    planning_limit=min(90.,timeout_s*.6)
    trajectory, plan = plan_request(control, targets, planning_limit)
    plan['planning_wall_clock_limit_s']=planning_limit
    if trajectory is None and plan['reason'] in ('initial_collision_requires_recovery','initial_clearance_requires_recovery'):
        return control._recovery_move(targets, timeout_s=timeout_s, max_actions=max_actions)
    actions = completed = 0
    terminal_tracking_errors = None
    terminal_tracking_actions = 0
    tracking = None
    placement = getattr(control, 'support_placement', None)
    step_limit = (TOOL_CONTACT_JOINT_STEP_LIMIT_RAD if hasattr(placement, 'workpieces')
                  else .01 if placement is not None else .04)
    budget = execution_budget(joint_action_estimate(trajectory,step_limit) if trajectory else 0, max_actions, timeout_s)
    if trajectory is None:
        control.stopped = control._public_plan(plan)
    else:
        for knot_index, target in enumerate(trajectory[1:], start=1):
            slots = [[r['joint_names'].index(n) for n in r['arm_joint_names']]
                     for r in control.world.scene.snapshot['robots']]
            tracking = {'knot_index': knot_index, 'actions_at_entry': actions,
                        'samples': deque(maxlen=13)}
            while not control.stopped:
                stop = exhausted(budget, started, actions, time.monotonic())
                if stop:
                    control.stopped = stop
                    break
                if terminal_tracking_errors is not None:
                    terminal_tracking_actions += 1
                control._action([array(q)[s] for q, s in zip(target, slots)][:len(control.roles)], control._grippers())
                actions += 1
                if control.adapter._terminal:
                    control.stopped = {'reason': 'official_episode_ended'}
                    break
                if control.stopped:
                    break
                actual = control.world.positions()
                tracking['samples'].append((actions, [
                    abs(array(a)[s]-array(b)[s]).copy()
                    for a, b, s in zip(actual[:len(control.roles)], target, slots)]))
                joint_ready = all(np.max(abs(array(a)[s]-array(b)[s])) < .008
                                  for a, b, s in zip(actual[:len(control.roles)], target, slots))
                terminal_errors = ({role: pose_error(control.tcp(control.roles.index(role)), array(path[-1]))
                                    for role, path in targets.items()}
                                   if knot_index == len(trajectory)-1 else {})
                terminal_ready = all(e['position_m'] < position_tolerance and e['orientation_rad'] < .03
                                     for e in terminal_errors.values())
                if joint_ready and terminal_ready:
                    completed += 1
                    break
                stalled = tracking_progress_stop(control, targets, tracking, actions, len(trajectory)-1)
                if stalled is not None:
                    control._halt('tracking_progress_stalled', stalled, preserve_gripper_targets=True)
                    break
                if joint_ready and terminal_tracking_errors is None:
                    terminal_tracking_errors = terminal_errors
            if control.stopped:
                break
    errors = {role: pose_error(control.tcp(control.roles.index(role)), array(path[-1])) for role, path in targets.items()}
    reached = bool(trajectory is not None and completed == len(trajectory)-1 and control.stopped is None
                   and all(e['position_m'] < position_tolerance and e['orientation_rad'] < .03 for e in errors.values()))
    if trajectory is not None and not reached and control.stopped is None:
        control.stopped = {'reason': 'measured_final_pose_outside_tolerance'}
    adjustments = []
    if terminal_tracking_errors is not None:
        adjustments.append({'kind': 'terminal_tcp_tracking',
            'reason': 'joint_tolerance_met_but_final_tcp_outside_tolerance',
            'initial_tcp_error': terminal_tracking_errors,
            'additional_native_actions': terminal_tracking_actions,
            'position_tolerance_m': position_tolerance, 'orientation_tolerance_rad': .03,
            'target_and_execution_limits': 'unchanged'})
    if actions and hasattr(placement, 'workpieces'):
        adjustments.append({'kind': 'bounded_tool_contact_motion_rate',
            'role': placement.role, 'joint_step_limit_rad': step_limit,
            'previous_joint_step_limit_rad': .005, 'applied_native_actions': actions,
            'scope': 'explicit_tool_contact_only; all_arm_increments_bounded',
            'native_drives_and_physics': 'unchanged',
            'safety_checks': 'unchanged_per_physics_step_collision_contact_path_and_attachment_checks',
            'faster_completion_or_stable_grasp_guaranteed': False})
    receipt = {'schema_version': 'openeta.robodojo_checked_motion.v1', 'reached_target': reached,
        'stop': control.stopped, 'completed_knots': completed, 'planned_knots': max(0, len(trajectory)-1) if trajectory else 0,
        'native_actions': actions, 'physics_steps': control.physics_steps,
        'joint_step_limit_rad': step_limit, 'joint_step_limited_actions': control.joint_step_limited_actions,
        'unselected_effectors': 'held_at_initial_measured_joint_targets',
        'elapsed_s': time.monotonic()-started, 'final_tcp_error': errors, 'budget':budget,
        'planning': control._public_plan(plan),
        'host_adjustments': adjustments,
        'execution': 'preflighted_joint_trajectory_with_per_physics_step_recheck',
        'collision_policy': control.world.public_policy(),
        'collision_guarantee': 'stationary_world_joint_segment_certificate_with_discrete_physics_monitor'}
    if tracking is not None and (control.stopped or {}).get('reason') in ('motion_budget_exhausted', 'tracking_progress_stalled'):
        receipt['tracking_diagnostics'] = tracking_feedback(control, targets, tracking, actions,
                                                          len(trajectory)-1)
        if control.stopped['reason'] == 'tracking_progress_stalled':
            receipt['tracking_diagnostics']['guidance'] = TRACKING_STALL_GUIDANCE
            receipt['tracking_diagnostics']['execution_policy_changed'] = True
            receipt['host_adjustments'].append({
                'kind': 'early_tracking_stop', 'reason': 'measured_joint_progress_below_guard',
                'waypoints_skipped': 0, 'target_and_precision': 'unchanged',
                'automatic_spatial_replanning': False, 'gripper_hold': 'preserve_previous_targets'})
    control.receipt = receipt
    return receipt


def tracking_feedback(control, targets, tracking, actions, planned_knots):
    """Measured robot tracking only; no contact-cause or object-state inference."""
    samples = tracking['samples']
    roles = {}
    if samples:
        first_action, first = samples[0]
        last_action, last = samples[-1]
        for i, role in enumerate(control.roles):
            errors = last[i]
            largest = int(np.argmax(errors))
            names = control.world.scene.snapshot['robots'][i]['arm_joint_names']
            roles[role] = {
                'selected': role in targets,
                'largest_error_joint': names[largest],
                'current_max_joint_error_rad': float(errors[largest]),
                'joint_tolerance_met': bool(np.max(errors) < .008),
                'recent_window_actions': last_action-first_action,
                'window_initial_max_joint_error_rad': float(np.max(first[i])),
                'max_joint_error_reduction_rad': float(np.max(first[i])-np.max(errors)),
            }
    return {
        'scope': 'measured_robot_joints_vs_current_planned_knot',
        'active_knot_index': tracking['knot_index'],
        'planned_knots': planned_knots,
        'active_knot_is_final': tracking['knot_index'] == planned_knots,
        'native_actions_on_active_knot': actions-tracking['actions_at_entry'],
        'joint_tolerance_rad': .008,
        'roles': roles,
        'sample_status': 'available' if samples else 'no_completed_tracking_sample',
        'guidance': ('Execution budget expired before tracking completed. final_tcp_error refers to the '
                     'requested endpoint; joint errors here refer to the current intermediate knot. '
                     'A positive reduction means the maximum joint error decreased over the reported '
                     'window, not guaranteed future convergence. Check the current images and contact '
                     'alignment before retrying or changing the approach. This does not establish '
                     'contact obstruction, torque saturation, or IK unreachability.'),
        'execution_policy_changed': False,
    }


TRACKING_STALL_GUIDANCE = (
    'Motion stopped early because a joint remained outside the tracking tolerance and made '
    'less than 0.0005 rad progress over 12 native actions, after at least 24 actions at the '
    'same planned knot. No waypoint was skipped and no alternate path was executed. '
    'Inspect current public images and robot state before choosing a retry or another approach. '
    'This observation does not establish contact obstruction, torque saturation or IK '
    'unreachability; it does not prove that longer tracking could never converge.')


def tracking_progress_stop(control, targets, tracking, actions, planned_knots):
    """Bound futile tracking using robot measurements; never relax/advance a knot."""
    samples = tracking['samples']
    if actions-tracking['actions_at_entry'] < 24 or len(samples) < 13:
        return None
    first_action, first = samples[0]
    last_action, last = samples[-1]
    if last_action-first_action != 12:
        return None
    stalled_roles = []
    for i, role in enumerate(control.roles):
        # Require every measured maximum in the window to remain outside the
        # existing gate, and use the best progress in that window. This avoids
        # calling a recovered/oscillating sample a stationary error plateau.
        maxima = np.asarray([np.max(values[i]) for _, values in samples])
        if (np.isfinite(maxima).all() and np.all(maxima >= .008)
                and float(maxima[0]-np.min(maxima)) < .0005):
            stalled_roles.append(role)
    if not stalled_roles:
        return None
    feedback = tracking_feedback(control, targets, tracking, actions, planned_knots)
    feedback.update(guidance=TRACKING_STALL_GUIDANCE, execution_policy_changed=True)
    return {'status': 'unknown', 'stalled_roles': stalled_roles,
        'minimum_actions_on_knot': 24, 'window_native_actions': 12,
        'minimum_window_progress_rad': .0005,
        'tracking': feedback,
        'guidance': TRACKING_STALL_GUIDANCE,
        'simulation_extra_steps': 0, 'target_and_precision': 'unchanged',
        'automatic_spatial_replanning': False}
