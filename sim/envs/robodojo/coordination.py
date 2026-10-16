"""Bounded concurrent/offset/serial scheduling of independent joint paths.

The scene checks every combined piecewise-linear segment, including start/end
holds and conditional attachments. Seconds describe a nominal common phase
clock; execution may only slow that clock for both arms together. External
moving actors require a different checker and are not admitted by this profile.
"""
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass
class JointTrack:
    role: str
    index: int
    times: np.ndarray
    positions: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        if (self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0
                or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0)
                or self.positions.ndim != 2 or len(self.positions) != len(self.times)
                or not np.isfinite(self.positions).all()):
            raise ValueError('Invalid timed joint track')

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, phase):
        return np.array([np.interp(phase, self.times, self.positions[:, j])
                         for j in range(self.positions.shape[1])])


def participant(scene, identifier, tracks):
    for index, item in enumerate(scene.items):
        if item['id'] != identifier:
            continue
        robot = item['robot_index']
        track = next((t for t in tracks if t.index == robot), None)
        value = {'geometry_id':f'geometry_{index}'}
        if track:
            attached = str(item['body']).startswith('held_geometry_')
            value.update(action_id=f'{track.role}_move', role=track.role,
                         kind='attached_object' if attached else 'robot',
                         part='held_object' if attached else item['body'])
        else:
            value['kind'] = 'scene_geometry'
        return value
    return {'kind':'unknown_geometry'}


def public_conflict(scene, check, tracks, window):
    detail = check.get('detail', check)
    contacts = detail.get('contacts', [])
    pair = ([contacts[0]['a'], contacts[0]['b']] if contacts else detail.get('closest_pair'))
    return {'reason':check.get('reason', 'collision_or_clearance'),
            'time_window_s':list(map(float, window)),
            'time_window_basis':'nominal_segment_containing_failed_check',
            'participants':[participant(scene, name, tracks) for name in pair] if pair else [],
            'check_status':check.get('status', 'blocked')}


def combined_state(start, tracks, delays, phase):
    result = [np.array(q, dtype=float, copy=True) for q in start]
    for track, delay in zip(tracks, delays):
        # Positive delay means q(t-delay), clamped to the initial/final state.
        result[track.index] = track.sample(phase-delay)
    return result


def schedule_knots(tracks, delays):
    return sorted({0., *[round(float(t+d), 12) for track,d in zip(tracks,delays) for t in track.times]})


def validate_schedule(scene, start, tracks, delays, allowed_pairs, deadline):
    times = schedule_knots(tracks, delays)
    for lo, hi in zip(times, times[1:]):
        if time.monotonic() >= deadline:
            return {'status':'unknown', 'reason':'coordination_search_budget'}
        check = scene.check_path(combined_state(start, tracks, delays, lo),
                                 combined_state(start, tracks, delays, hi),
                                 allowed_pairs=allowed_pairs,
                                 timeout_s=min(2., max(0., deadline-time.monotonic())), max_nodes=512)
        if check['status'] != 'clear':
            return {'status':check['status'], 'conflict':public_conflict(scene, check, tracks, (lo, hi))}
    return {'status':'clear'}


def coordinate(scene, start, tracks, *, allowed_pairs=(), max_delay_s=5., delay_step_s=.1,
               timeout_s=20., spatial_replanner=None):
    """Return a nominal schedule and evidence; finite failure proves no impossibility."""
    if len(tracks) != 2 or len({t.index for t in tracks}) != 2 or len({t.role for t in tracks}) != 2:
        raise ValueError('Independent coordination requires exactly two distinct arms')
    if (not all(math.isfinite(v) for v in (max_delay_s, delay_step_s, timeout_s))
            or max_delay_s < 0 or delay_step_s <= 0 or max_delay_s/delay_step_s > 1000):
        raise ValueError('Invalid bounded retiming search')
    if len(start) != len(scene.snapshot['robots']):
        raise ValueError('Missing articulation state')
    for track in tracks:
        if not 0 <= track.index < len(start) or not np.allclose(track.positions[0], start[track.index], atol=1e-9):
            raise ValueError('Track does not start at measured joint state')
        robot = scene.snapshot['robots'][track.index]
        if track.positions.shape[1] != len(robot['joint_names']):
            raise ValueError('Track joint identity/shape mismatch')
        held = [i for i,n in enumerate(robot['joint_names']) if n not in robot['arm_joint_names']]
        if held and not np.allclose(track.positions[:, held], np.array(start[track.index])[held], atol=1e-9):
            raise ValueError('Coordination cannot change gripper intent')
    result = {'schema':'openeta.bimanual_coordination.v1', 'status':'rejected',
              'attempts':{'concurrent':0, 'temporal_shift':0, 'serialize':0},
              'adjustments':[], 'search_exhaustion_is_infeasibility_proof':False,
              'phase_clock':'nominal_seconds_with_common_tracking_holds',
              'world_assumption':'stationary_except_selected_arms_and_conditional_attachments'}
    if scene.snapshot.get('closed_chain_constraints'):
        return {**result, 'reason':'shared_body_requires_coupled_planning'}
    deadline = time.monotonic()+max(0., timeout_s)
    if time.monotonic() >= deadline:
        return {**result, 'reason':'coordination_search_budget'}
    for label, q in [('initial', start), ('final', combined_state(start, tracks, [0.,0.],
                                                               max(t.duration for t in tracks)))]:
        check = scene.check_path(q, q, allowed_pairs=allowed_pairs,
                                 timeout_s=min(2., max(0., deadline-time.monotonic())), max_nodes=512)
        if check['status'] != 'clear':
            phase = 0. if label == 'initial' else max(t.duration for t in tracks)
            return {**result, 'reason':f'candidate_{label}_configuration_not_certified',
                    'conflict':public_conflict(scene, check, tracks, (phase,phase))}

    def attempt(mode, delays):
        result['attempts'][mode] += 1
        checked = validate_schedule(scene, start, tracks, delays, allowed_pairs, deadline)
        if checked['status'] != 'clear':
            result.setdefault('conflict', checked.get('conflict'))
            result['last_check_status'] = checked['status']
            return None
        schedule = [{'action_id':f'{t.role}_move', 'role':t.role, 'start_s':float(d),
                     'duration_s':t.duration, 'end_s':float(d+t.duration)} for t,d in zip(tracks,delays)]
        return {**result, 'status':'safe' if mode == 'concurrent' else 'repaired',
                'mode':mode, 'schedule':schedule, 'delays_s':list(map(float, delays)),
                'selected_schedule_check':'clear',
                'makespan_s':max(d+t.duration for d,t in zip(delays,tracks)),
                'adjustments':([] if mode == 'concurrent' else
                    [{'kind':mode, 'schedule':schedule,
                      'reason':'independent_motion_collision_coordination',
                      'optimality':'first_verified_grid_delay' if mode == 'temporal_shift'
                                    else 'shortest_nominal_makespan_tested_order'}])}

    answer = attempt('concurrent', [0.,0.])
    if answer:
        return answer
    count = int(math.floor(max_delay_s/delay_step_s+1e-9))
    retime_deadline = (min(deadline,time.monotonic()+max(0.,(deadline-time.monotonic())*.4))
                       if spatial_replanner else deadline)
    for step in range(1, count+1):
        if time.monotonic() >= retime_deadline:
            result['retiming_search_truncated_for_fallbacks']=True
            break
        delay = step*delay_step_s
        for delayed in (1, 0):
            # No overlap belongs to the serialization stage, not retiming.
            if delay >= tracks[1-delayed].duration:
                continue
            if time.monotonic() >= deadline:
                return {**result, 'reason':'coordination_search_budget'}
            shifts = [0.,0.]; shifts[delayed] = delay
            answer = attempt('temporal_shift', shifts)
            if answer:
                return answer
    if spatial_replanner and time.monotonic()<deadline:
        result['attempts']['spatial_replanning']=1
        replacements, detail=spatial_replanner(tracks,max(0.,deadline-time.monotonic()))
        result['spatial_replanning']=detail
        if replacements is not None:
            # Recheck all replacement segments with the complete scene. The
            # callback owns waypoint preservation; it may never change endpoints.
            if (len(replacements)!=len(tracks) or any(a.role!=b.role or a.index!=b.index
                    or not np.allclose(a.positions[0],b.positions[0],atol=1e-8)
                    or not np.allclose(a.positions[-1],b.positions[-1],atol=1e-8)
                    for a,b in zip(tracks,replacements))):
                # Different IK solutions may preserve TCP endpoints but need an
                # explicit endpoint contract from the trusted planner.
                same_roles=len(replacements)==len(tracks) and all(a.role==b.role and a.index==b.index
                    and np.allclose(a.positions[0],b.positions[0],atol=1e-8) for a,b in zip(tracks,replacements))
                if not same_roles or detail.get('preserves')!='every_agent_endpoint_and_waypoint':
                    raise ValueError('Spatial replanner changed action identities or endpoint contract')
            checked=validate_schedule(scene,start,replacements,[0.,0.],allowed_pairs,deadline)
            if checked['status']=='clear':
                tracks[:]=replacements
                schedule=[{'action_id':f'{t.role}_move','role':t.role,'start_s':0.,
                           'duration_s':t.duration,'end_s':t.duration} for t in tracks]
                return {**result,'status':'repaired','mode':'spatial_replanning','schedule':schedule,
                        'delays_s':[0.,0.],'makespan_s':max(t.duration for t in tracks),
                        'selected_schedule_check':'clear',
                        'adjustments':[{'kind':'spatial_replanning','reason':'temporal_repair_not_certified',
                                        'planning':detail}]}
    # Both orders have the same nominal duration; deterministic tie-break.
    for shifts in ([0.,tracks[0].duration], [tracks[1].duration,0.]):
        if time.monotonic() >= deadline:
            return {**result, 'reason':'coordination_search_budget'}
        answer = attempt('serialize', shifts)
        if answer:
            return answer
    return {**result, 'reason':'no_certified_schedule_within_search',
            'guidance':'Change an endpoint or waypoint, or clear workspace in a separate request.'}


def execution_samples(start, tracks, delays, period_s):
    if not math.isfinite(period_s) or period_s <= 0:
        raise ValueError('Invalid native control period')
    knots = schedule_knots(tracks, delays)
    # Retain all path corners; never shortcut a checked piecewise-linear track.
    for lo,hi in zip(knots,knots[1:]):
        for phase in np.linspace(lo, hi, max(1, math.ceil((hi-lo)/period_s-1e-9))+1)[1:]:
            yield float(phase), combined_state(start, tracks, delays, float(phase))
