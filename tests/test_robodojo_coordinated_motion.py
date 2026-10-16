from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.checked_control import CheckedController
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.scene_geometry import forward_kinematics
from sim.envs.robodojo.x5_calibration import link_to_tcp, tcp_to_link
from test_robodojo_joint_planning import dual_crossing


def controller(*, lag_once=False):
    snapshot = deepcopy(dual_crossing().snapshot)
    for i,r in enumerate(snapshot['robots']):
        r['role'] = ('left','right')[i]
        r['body_frames']['link6'] = r['body_frames'].pop('tool')
        r['joints'][0]['child'] = 'link6'
    for item in snapshot['colliders']:
        if item['body'] == 'tool':
            item['body'] = 'link6'
    scene = CollisionScene(snapshot)
    positions = [np.array([-.3]),np.array([-.3])]
    c = CheckedController.__new__(CheckedController)
    c.roles = ['left','right']; c.allowed_pairs = []; c.contact_grants = {}
    c.world = SimpleNamespace(scene=scene, refresh=lambda:None, positions=lambda:[p.copy() for p in positions],
                              public_policy=lambda:{}, table_id='world')
    c.native = SimpleNamespace(sim=SimpleNamespace(physics_dt=.004,
                               cfg=SimpleNamespace(sim=SimpleNamespace(dt=.004))),
                               obs_manager=SimpleNamespace(collect_interval=10))
    c.native.robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1,3))))
    c.adapter = SimpleNamespace(_terminal=False)
    c.tcp = lambda i:link_to_tcp(forward_kinematics(snapshot['robots'][i], positions[i])['link6'])
    c._pose_candidates = lambda i, pose, q:([np.array([tcp_to_link(pose)[i,3]])], {'count':1})
    c._grippers = lambda:[1.,1.]
    history = []
    def action(targets, grippers):
        history.append([np.array(q).copy() for q in targets])
        c.physics_steps += c.native.obs_manager.collect_interval
        if lag_once and len(history) == 1:
            return
        # The integration must send only jointly checked motion segments.
        assert scene.check_path(positions,targets)['status'] == 'clear'
        for i,q in enumerate(targets):
            positions[i] = np.array(q).copy()
    c._action = action
    c._halt = lambda reason, **kwargs:setattr(c, 'stopped', {'reason':reason})
    return c, history


def goals(c, end=.3):
    result = {}
    for i,role in enumerate(c.roles):
        pose = c.tcp(i)
        pose[i,3] += end+.3
        result[role] = [pose]
    return result


def test_default_backend_stays_synchronized_but_independent_mode_executes_repaired_schedule():
    c, history = controller()
    result = c.move(goals(c), coordination='independent', max_actions=180)
    assert result['reached_target'] and result['status'] == 'completed_with_adjustment'
    assert result['coordination']['mode'] == 'temporal_shift'
    assert result['actual_timing']['left']['started_at_sim_s'] == 0.
    assert result['actual_timing']['right']['started_at_sim_s'] > 0.
    assert result['native_actions'] == len(history)
    assert result['physics_steps'] == len(history)*10


def test_infeasible_final_candidate_or_execution_budget_rejects_before_physics():
    c, history = controller()
    result = c.move(goals(c, end=0.), coordination='independent')
    assert result['status'] == 'rejected' and history == []
    assert result['stop']['physics_advanced'] is False
    c, history = controller()
    result = c.move(goals(c), coordination='independent', max_actions=1)
    assert result['status'] == 'rejected' and history == []
    assert result['coordination']['reason'] == 'nominal_schedule_exceeds_execution_budget'


def test_tracking_lag_holds_common_schedule_and_is_reported():
    c, history = controller(lag_once=True)
    result = c.move(goals(c), coordination='independent')
    assert result['reached_target']
    assert all(np.allclose(a,b) for a,b in zip(history[0],history[1]))
    assert result['host_adjustments'][-1]['kind'] == 'common_phase_tracking_hold'
    assert result['host_adjustments'][-1]['additional_native_actions'] == 1


def test_later_unreachable_waypoint_never_executes_earlier_motion():
    c, history = controller()
    target = goals(c, end=-.2)
    extra = target['left'][0].copy(); extra[0,3] += .2
    target['left'].append(extra)
    original = c._pose_candidates
    c._pose_candidates = lambda i,pose,q: ([], {'count':0}) if i == 0 and tcp_to_link(pose)[0,3] > -.19 else original(i,pose,q)
    result = c.move(target, coordination='independent')
    assert result['status'] == 'rejected' and history == []
    assert result['coordination']['waypoint'] == 1


def test_long_safe_schedule_uses_derived_budget_instead_of_180_action_ceiling():
    c, history = controller()
    # Slow nominal joint motion creates >180 samples without changing geometry.
    target=goals(c)
    c.native.obs_manager.collect_interval=2
    result=c.move(target,coordination='independent')
    assert result['budget']['required_nominal_actions']>180
    assert result['budget']['action_limit']>result['budget']['required_nominal_actions']
    assert result['reached_target'] and len(history)>180


def test_interior_ik_failure_rejects_without_spatial_replanning(monkeypatch):
    import sim.envs.robodojo.spatial_replanning as spatial
    def forbidden(*args, **kwargs):
        raise AssertionError('Host must never replace the requested route')
    monkeypatch.setattr(spatial, 'replan_tracks', forbidden)
    c, history=controller()
    original=c._pose_candidates
    c._pose_candidates=lambda i,pose,q: ([],{'count':0}) if i==0 and -.27<tcp_to_link(pose)[0,3]<-.23 else original(i,pose,q)
    result=c.move(goals(c,end=-.18),coordination='independent')
    assert result['status']=='rejected' and not history
    assert result['coordination']['reason']=='independent_cartesian_candidate_not_certified'
    assert result['coordination']['automatic_spatial_replanning'] is False
    assert result['native_actions']==0


def test_requested_precision_holds_both_arms_at_final_phase():
    c,history=controller();targets=goals(c);original_tcp=c.tcp
    c.position_tolerance_m=.0005
    def tcp(index):
        pose=original_tcp(index).copy()
        final=bool(history) and all(abs(float(q[0])-.3)<1e-8 for q in history[-1])
        repeated=final and len(history)>1 and all(np.array_equal(a,b) for a,b in zip(history[-1],history[-2]))
        if final and not repeated:pose[2,3]+=.001
        return pose
    c.tcp=tcp
    result=c.move(targets,coordination='independent',max_actions=180)
    assert result['reached_target']
    hold=result['host_adjustments'][-1]
    assert hold['kind']=='common_phase_tracking_hold' and hold['additional_native_actions']==1
    assert 'requested_tcp' in hold['reason']
    assert all(np.array_equal(a,b) for a,b in zip(history[-1],history[-2]))


def test_candidate_edges_share_mesh_budget_and_respect_remaining_deadline(monkeypatch):
    import time
    from sim.envs.robodojo.coordinated_motion import candidate_tracks
    c, history = controller()
    scene = c.world.scene; original = scene.check_path; budgets = []
    def checked(*args, **kwargs):
        budgets.append(kwargs['timeout_s'])
        assert kwargs['max_nodes'] == 512
        return original(*args, **kwargs)
    monkeypatch.setattr(scene, 'check_path', checked)
    for remaining in (10., .5):
        budgets.clear()
        tracks, failure = candidate_tracks(c, goals(c, end=-.25), scene,
            c.world.positions(), .04, time.monotonic()+remaining)
        assert tracks and failure is None and budgets
        assert max(budgets) <= min(4., remaining)
        if remaining == 10.:
            assert min(budgets) == 4.
    assert not history


def horizon_controller(*, lag_once=False):
    from sim.envs.robodojo.execution_horizon import ExecutionHorizon
    c, history = controller(lag_once=lag_once)
    h = c._execution_horizon = ExecutionHorizon(.08, physics_dt=.004, steps_per_action=10)
    original = c._action
    def action(*args):
        assert not h.exhausted()
        original(*args)
        for _ in range(10):
            h.after_step()
    c._action = action
    return c, history, h


def test_explicit_horizon_executes_checked_prefix_and_keeps_delayed_arm_unstarted():
    c, history, h = horizon_controller()
    result = c.move(goals(c), coordination='independent', max_actions=h.cap())
    h.decorate(c, result)
    assert result['native_actions'] == len(history) == 2
    assert result['physics_steps'] == h.used_steps == 20
    assert result['status'] == 'interrupted' and not result['reached_target']
    assert result['stop']['reason'] == 'requested_execution_horizon_reached'
    assert result['actual_timing']['right']['started_at_sim_s'] is None
    window = result['coordination']['execution_window']
    assert window['complete_schedule_prechecked'] and not window['automatic_resume']
    assert not window['target_changed'] and window['full_nominal_duration_s'] > .08
    assert result['completed_knots'] < result['planned_knots']
    assert all(np.allclose(q[1], [-.3]) for q in history)


def test_tracking_holds_consume_the_same_physical_horizon_without_extending_it():
    c, history, h = horizon_controller(lag_once=True)
    result = c.move(goals(c), coordination='independent', max_actions=h.cap())
    h.decorate(c, result)
    assert len(history) == result['native_actions'] == 2 and h.used_steps == 20
    assert result['completed_knots'] == 1
    assert all(np.allclose(a,b) for a,b in zip(history[0],history[1]))
    assert result['execution_horizon']['actual_duration_s'] == .08
    assert result['stop']['reason'] == 'requested_execution_horizon_reached'


def test_horizon_does_not_override_collision_or_a_stricter_other_budget():
    for target_end, max_actions in [(0., 2), (.3, 1)]:
        c, history, h = horizon_controller()
        result = c.move(goals(c, end=target_end), coordination='independent', max_actions=max_actions)
        h.decorate(c, result)
        assert result['status'] == 'rejected' and not history and h.used_steps == 0
        assert 'execution_window' not in result['coordination']
    c, history, h = horizon_controller()
    original = c._action
    def stop_early(*args):
        original(*args)
        c.stopped = {'reason':'measured_collision_after_physics'}
    c._action = stop_early
    result = c.move(goals(c), coordination='independent', max_actions=h.cap())
    h.decorate(c, result)
    assert len(history) == 1 and h.used_steps == 10
    assert result['stop']['reason'] == 'measured_collision_after_physics'
