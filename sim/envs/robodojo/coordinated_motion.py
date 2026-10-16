"""Independent dual-X5 motion integration; existing synchronized path is separate."""
import math
import time

import numpy as np

from .attached_geometry import attached_scene
from .coordination import JointTrack, coordinate, execution_samples
from .motion_planning import validate_motion_targets, CARTESIAN_EDGE_CHECK_TIMEOUT_S
from .motion_budget import execution_budget, exhausted, HARD_ACTION_LIMIT


def native_period(control):
    # native.sim is IsaacRLEnv, not SimulationCfg. Use the same accessor as
    # CustomDirectRLEnv.sim_step when it updates the physical scene.
    dt = float(control.native.sim.physics_dt)
    count = max(1, int(control.native.obs_manager.collect_interval))
    if not math.isfinite(dt) or dt <= 0 or count > 1000:
        raise ValueError('Unverified native control timebase')
    return dt, count, dt*count


def candidate_tracks(control, targets, scene, start, period, deadline):
    from .checked_control import cartesian_knots
    selected = [control.roles.index(role) for role in targets]
    # Ignore inter-selected-arm pairs only while producing independent candidates.
    # Every acceptance check below restores the complete scene and attached objects.
    candidate_only_pairs = [(a['id'], b['id']) for i,a in enumerate(scene.items) for b in scene.items[:i]
                            if a['robot_index'] in selected and b['robot_index'] in selected
                            and a['robot_index'] != b['robot_index']]
    tracks = []
    for index in selected:
        role = control.roles[index]
        robot = scene.snapshot['robots'][index]
        slots = [robot['joint_names'].index(n) for n in robot['arm_joint_names']]
        cursor = [np.array(q, copy=True) for q in start]
        poses = [np.array(cursor[index], copy=True)]
        times = [0.]
        previous = control.tcp(index)
        for waypoint, endpoint in enumerate(targets[role]):
            for knot_index, knot in enumerate(cartesian_knots(previous, np.asarray(endpoint))):
                if time.monotonic() >= deadline:
                    return None, {'reason':'candidate_generation_budget', 'role':role, 'waypoint':waypoint}
                candidates, diagnostics = control._pose_candidates(index, knot, cursor[index])
                chosen = None
                last = None
                for q in candidates:
                    if time.monotonic() >= deadline:
                        break
                    trial = [a.copy() for a in cursor]
                    trial[index][slots] = q
                    last = scene.check_path(cursor, trial,
                        allowed_pairs=[*control.allowed_pairs, *candidate_only_pairs],
                        timeout_s=min(CARTESIAN_EDGE_CHECK_TIMEOUT_S,max(0.,deadline-time.monotonic())), max_nodes=512)
                    if last['status'] == 'clear':
                        chosen = trial
                        break
                if chosen is None:
                    return None, {'reason':'independent_cartesian_candidate_not_certified',
                        'role':role, 'waypoint':waypoint, 'ik':diagnostics,
                        'phase':'cartesian_interpolation', 'knot_index':knot_index,
                        'failed_tcp_xyz':knot[:3,3].tolist(),
                        'detail':control._public_plan(last),
                        'search_exhaustion_is_infeasibility_proof':False}
                # Nominal 0.5 rad/s joint speed, quantized to native action periods.
                duration = max(period, math.ceil(float(np.max(abs(chosen[index]-cursor[index])))/.5/period)*period)
                times.append(round(times[-1]+duration, 12))
                poses.append(chosen[index].copy())
                cursor = chosen
            previous = np.asarray(endpoint)
        tracks.append(JointTrack(role, index, times, poses))
    return tracks, None


def execute_independent(control, targets, *, timeout_s=180., max_actions=None):
    from .checked_control import pose_error
    validate_motion_targets(control, targets)
    from .motion_precision import tolerance
    position_tolerance = tolerance(getattr(control, 'position_tolerance_m', .003))
    if len(control.roles) != 2 or set(targets) != set(control.roles):
        raise ValueError('Independent coordination requires both configured target arms')
    dt, native_steps, period = native_period(control)
    control.stopped = None; control.physics_steps = 0; control.last_step = None
    control.operation = 'move'; control.attachment_baseline = {}; control.recovery = None
    control.joint_step_limited_actions = 0
    began = time.monotonic()
    planning_limit=min(90.,timeout_s*.6)
    deadline = began+planning_limit
    candidate_deadline=deadline-min(15.,planning_limit*.2)
    control.world.refresh()
    start = control.world.positions()
    scene, attachments = attached_scene(control.world, start, control.contact_grants)
    tracks = None
    rejection = None
    # Never independently schedule two wrists constrained by a shared body.
    if scene.snapshot.get('closed_chain_constraints'):
        rejection = {'reason':'shared_body_requires_coupled_planning'}
    else:
        initial = scene.check_path(start, start, allowed_pairs=control.allowed_pairs,
                                   timeout_s=min(2.,max(0.,deadline-time.monotonic())))
        if initial['status'] != 'clear':
            rejection = {'reason':'initial_clearance_requires_separate_recovery' if initial.get('reason')=='insufficient_clearance' else 'initial_collision_requires_separate_recovery',
                         'detail':control._public_plan(initial),
                         'guidance':'Resolve the initial boundary with a short orientation-preserving single-arm retreat before independent scheduling.'}
        else:
            tracks, rejection = candidate_tracks(control, targets, scene, start, period, candidate_deadline)
    if rejection:
        coordination = {'schema':'openeta.bimanual_coordination.v1', 'status':'rejected',
                        **rejection, 'adjustments':[]}
    else:
        coordination = coordinate(scene, start, tracks, allowed_pairs=control.allowed_pairs,
            timeout_s=max(0.,deadline-time.monotonic()), max_delay_s=5.,
            delay_step_s=math.ceil(.1/period)*period)
    coordination['automatic_spatial_replanning'] = False
    if rejection:
        coordination['guidance'] = 'The requested Cartesian path failed preview; no alternate spatial route was searched or executed. Inspect the evidence and choose new targets or waypoints.'
    coordination['planning_wall_clock_limit_s']=planning_limit
    coordination['candidate_generation_wall_clock_limit_s']=candidate_deadline-began
    coordination['cartesian_edge_check_time_limit_s']=CARTESIAN_EDGE_CHECK_TIMEOUT_S
    samples = []
    if coordination['status'] != 'rejected':
        # Bound allocation and reject an over-budget schedule before any motion.
        for sample in execution_samples(start, tracks, coordination['delays_s'], period):
            samples.append(sample)
            if len(samples) > HARD_ACTION_LIMIT:
                coordination.update(status='rejected', reason='nominal_schedule_exceeds_execution_budget')
                break
    budget = execution_budget(len(samples), max_actions, timeout_s)
    horizon = getattr(control, '_execution_horizon', None)
    # A requested observation ceiling permits a checked prefix, like single-arm
    # motion. Keep hard allocation limits and non-horizon budget rejection.
    prefix_requested = (horizon is not None and horizon.actions == budget['action_limit'])
    if coordination['status'] != 'rejected' and len(samples) > budget['action_limit']:
        if prefix_requested:
            coordination['execution_window'] = {
                'policy':'checked_common_schedule_prefix_then_observe',
                'full_nominal_duration_s':coordination['makespan_s'],
                'requested_max_duration_s':horizon.requested,
                'maximum_native_actions':horizon.actions,
                'complete_schedule_prechecked':True,
                'target_changed':False, 'automatic_resume':False,
                'guidance':'The full checked schedule exceeds this observation window. '
                           'One or both arms may remain incomplete or unstarted; inspect '
                           'actual_timing, measured poses and the stop reason before another request.'}
        else:
            coordination.update(status='rejected', reason='nominal_schedule_exceeds_execution_budget')
    if coordination.get('reason') == 'nominal_schedule_exceeds_execution_budget':
        coordination['budget'] = budget
    if coordination['status'] == 'rejected':
        control.stopped = {'reason':coordination['reason'], 'physics_advanced':False}
    actions = completed = tracking_holds = 0
    timings = {role:{'started_at_sim_s':None,'completed_at_sim_s':None} for role in targets}
    slots = [[r['joint_names'].index(n) for n in r['arm_joint_names']]
             for r in scene.snapshot['robots']]
    for phase, target in samples if not control.stopped else []:
        while not control.stopped:
            stop = exhausted(budget, began, actions, time.monotonic())
            if stop:
                control.stopped = stop
                break
            for track in tracks:
                if (timings[track.role]['started_at_sim_s'] is None
                        and np.max(abs(target[track.index]-start[track.index])) > 1e-9):
                    timings[track.role]['started_at_sim_s'] = control.physics_steps*dt
            steps_before = control.physics_steps
            control._action([q[s] for q,s in zip(target,slots)][:len(control.roles)], control._grippers())
            actions += 1
            if control.stopped:
                break
            if control.adapter._terminal:
                control.stopped = {'reason':'official_episode_ended'}
                break
            if control.physics_steps-steps_before != native_steps:
                control._halt('native_control_timebase_changed')
                break
            actual = control.world.positions()
            terminal_ready = True
            if position_tolerance < .003 and phase == samples[-1][0]:
                terminal_errors = [pose_error(control.tcp(control.roles.index(role)), np.asarray(path[-1]))
                                   for role,path in targets.items()]
                terminal_ready = all(e['position_m'] < position_tolerance and e['orientation_rad'] < .03
                                     for e in terminal_errors)
            if terminal_ready and all(np.max(abs(np.array(a)[s]-b[s])) < .008 for a,b,s in zip(actual[:len(control.roles)],target,slots)):
                completed += 1
                for track,row in zip(tracks,coordination['schedule']):
                    if phase+1e-9 >= row['end_s'] and timings[track.role]['completed_at_sim_s'] is None:
                        timings[track.role]['completed_at_sim_s'] = control.physics_steps*dt
                break
            # Pause the COMMON phase, not one arm's relative phase independently.
            tracking_holds += 1
        if control.stopped:
            break
    errors = {role:pose_error(control.tcp(control.roles.index(role)), np.asarray(path[-1]))
              for role,path in targets.items()}
    reached = bool(samples and completed == len(samples) and not control.stopped
                   and all(e['position_m'] < position_tolerance and e['orientation_rad'] < .03 for e in errors.values()))
    if not reached and control.stopped is None:
        control.stopped = {'reason':'measured_final_pose_outside_tolerance'}
    adjustments = list(coordination['adjustments']) if actions else []
    if tracking_holds:
        adjustments.append({'kind':'common_phase_tracking_hold', 'additional_native_actions':tracking_holds,
                            'additional_sim_s':tracking_holds*period,
                            'reason':'measured_joint_or_requested_tcp_tracking_not_yet_within_tolerance' if position_tolerance < .003 else 'measured_joint_tracking_not_yet_within_tolerance'})
    status = ('completed_with_adjustment' if adjustments else 'completed') if reached else (
        'interrupted' if actions else 'rejected')
    return {'schema_version':'openeta.robodojo_checked_motion.v1', 'status':status,
            'reached_target':reached, 'stop':control.stopped,
            'completed_knots':completed, 'planned_knots':len(samples),
            'native_actions':actions, 'physics_steps':control.physics_steps,
            'elapsed_s':time.monotonic()-began, 'elapsed_sim_s':control.physics_steps*dt,
            'budget':budget,
            'final_tcp_error':errors, 'coordination':coordination,
            'actual_timing':timings, 'host_adjustments':adjustments, 'attachments':attachments,
            'execution':'joint_schedule_with_common_phase_tracking_holds_and_physics_monitor',
            'collision_guarantee':'stationary_world_combined_joint_segments; discrete_physics_monitor',
            'collision_policy':control.world.public_policy()}
