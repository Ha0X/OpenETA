"""Final Cartesian convergence must use the remaining checked tracking budget."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.envs.robodojo import motion_planning


def controller(monkeypatch, *, roles=1, orientation=False, knots=1, stall=False,
               stop_at=None, terminal_at=None):
    q = [0.] * roles
    calls = []
    c = SimpleNamespace(roles=['left', 'right'][:roles], adapter=SimpleNamespace(_terminal=False))
    c.world = SimpleNamespace(positions=lambda: [np.array([v]) for v in q],
        scene=SimpleNamespace(snapshot={'robots': [
            {'joint_names': ['a'], 'arm_joint_names': ['a']} for _ in q]}), public_policy=lambda: {})

    def pose(value):
        t = np.eye(4)
        if orientation:
            t[:3, :3] = Rotation.from_rotvec([0, 0, value * 6.5]).as_matrix()
        else:
            t[0, 3] = value * .65
        return t

    c.tcp = lambda i: pose(q[i])
    c._grippers = lambda: [0.] * roles
    c._public_plan = lambda p: p

    def action(target, grippers):
        calls.append(target)
        c.physics_steps += 10
        for i, t in enumerate(target):
            residual = (.006 if stall or len(calls) == 1 else .002) if i == roles - 1 else 0.
            q[i] = float(t[0]) - residual
        if len(calls) == stop_at:
            c.stopped = {'reason': 'fixture_collision'}
        if len(calls) == terminal_at:
            c.adapter._terminal = True

    c._action = action
    trajectory = [[[v] for _ in q] for v in ([0., .02] if knots == 1 else [0., .01, .02])]
    monkeypatch.setattr(motion_planning, 'plan_request',
        lambda *a: (trajectory, {'status': 'clear', 'stages': []}))
    return c, {role: [pose(.02)] for role in c.roles}


@pytest.mark.parametrize('options', [{}, {'orientation': True}, {'roles': 2}])
def test_final_tcp_converges_after_joint_tolerance_and_reports_extra_action(monkeypatch, options):
    c, targets = controller(monkeypatch, **options)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    assert result['reached_target'] and result['native_actions'] == 2
    assert result['completed_knots'] == 1
    adjustment, = result['host_adjustments']
    assert adjustment['kind'] == 'terminal_tcp_tracking'
    assert adjustment['additional_native_actions'] == 1
    assert adjustment['target_and_execution_limits'] == 'unchanged'
    assert adjustment['position_tolerance_m'] == .003
    assert adjustment['orientation_tolerance_rad'] == .03


@pytest.mark.parametrize('budget,stall', [(0, False), (1, False), (4, True)])
def test_tracking_respects_explicit_action_budget(monkeypatch, budget, stall):
    c, targets = controller(monkeypatch, stall=stall)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=budget)
    assert not result['reached_target'] and result['native_actions'] == budget
    assert result['completed_knots'] == 0
    assert result['stop']['reason'] == 'motion_budget_exhausted'
    if budget:
        assert result['host_adjustments'][0]['additional_native_actions'] == budget - 1


@pytest.mark.parametrize('options,reason,actions', [
    ({'stop_at': 2}, 'fixture_collision', 2),
    ({'terminal_at': 1}, 'official_episode_ended', 1),
])
def test_guard_or_native_terminal_never_counts_a_completed_knot(monkeypatch, options, reason, actions):
    c, targets = controller(monkeypatch, **options)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    assert not result['reached_target'] and result['stop']['reason'] == reason
    assert result['native_actions'] == actions and result['completed_knots'] == 0


def test_intermediate_knots_do_not_require_the_final_tcp_target(monkeypatch):
    c, targets = controller(monkeypatch, knots=2)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    assert result['reached_target'] and result['native_actions'] == result['completed_knots'] == 2
    assert result['host_adjustments'] == []


def test_contact_rate_receipt_reports_effective_cap_without_bypassing_terminal_checks(monkeypatch):
    c, targets = controller(monkeypatch)
    c.support_placement = SimpleNamespace(workpieces={}, role='left')
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    assert result['reached_target'] and result['native_actions'] == 2
    assert result['joint_step_limit_rad'] == .01
    assert result['host_adjustments'][0]['kind'] == 'terminal_tcp_tracking'
    rate = result['host_adjustments'][-1]
    assert rate['kind'] == 'bounded_tool_contact_motion_rate'
    assert rate['role'] == 'left' and rate['applied_native_actions'] == 2
    assert rate['previous_joint_step_limit_rad'] == .005
    assert rate['joint_step_limit_rad'] == result['joint_step_limit_rad']
    assert not rate['faster_completion_or_stable_grasp_guaranteed']


def test_wall_clock_exhaustion_prevents_extra_tracking(monkeypatch):
    c, targets = controller(monkeypatch)
    clock = iter([0., 0., 11., 11.])
    monkeypatch.setattr(motion_planning.time, 'monotonic', lambda: next(clock))
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    assert not result['reached_target'] and result['native_actions'] == 1
    assert result['stop']['budget_kind'] == 'wall_clock'
    assert result['host_adjustments'][0]['additional_native_actions'] == 0


def test_intermediate_budget_feedback_reports_slow_progress_without_skipping_knot(monkeypatch):
    c, targets = controller(monkeypatch, roles=2, knots=2)
    actions = []
    def action(target, grippers):
        actions.append(target)
        c.physics_steps += 10
    c._action = action
    c.world.positions = lambda: [np.array([.01]), np.array([-.01 + len(actions)*.0001])]
    result = motion_planning.execute_request(c, {'left': targets['left']}, timeout_s=10., max_actions=20)
    assert result['native_actions'] == 20 and result['completed_knots'] == 0
    d = result['tracking_diagnostics']
    assert d['active_knot_index'] == 1 and not d['active_knot_is_final']
    assert d['native_actions_on_active_knot'] == 20
    assert d['roles']['left']['joint_tolerance_met']
    blocked = d['roles']['right']
    assert not blocked['selected'] and not blocked['joint_tolerance_met']
    assert blocked['largest_error_joint'] == 'a'
    assert blocked['recent_window_actions'] == 12
    assert blocked['current_max_joint_error_rad'] == pytest.approx(.018)
    assert blocked['max_joint_error_reduction_rad'] == pytest.approx(.0012)
    assert d['execution_policy_changed'] is False


def test_budget_before_first_action_has_no_invented_tracking_sample(monkeypatch):
    c, targets = controller(monkeypatch)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=0)
    assert result['native_actions'] == 0
    assert result['tracking_diagnostics']['sample_status'] == 'no_completed_tracking_sample'
    assert result['tracking_diagnostics']['roles'] == {}


def test_plateau_reported_as_zero_reduction_without_claiming_contact_cause(monkeypatch):
    c, targets = controller(monkeypatch, stall=True)
    result = motion_planning.execute_request(c, targets, timeout_s=10., max_actions=4)
    d = result['tracking_diagnostics']
    assert d['active_knot_is_final'] and d['roles']['left']['joint_tolerance_met']
    assert d['roles']['left']['max_joint_error_reduction_rad'] == 0.
    assert not result['reached_target']
    assert 'does not establish' in d['guidance']


def test_joint_plateau_stops_early_without_skipping_or_relaxing(monkeypatch):
    c, targets = controller(monkeypatch, knots=2)
    actions=[];halts=[]
    def action(target, grippers):
        actions.append(target)
        c.physics_steps+=10
    def halt(reason, detail, *, preserve_gripper_targets=False):
        halts.append(preserve_gripper_targets)
        c.stopped={'reason':reason,'detail':detail}
    c._action=action;c._halt=halt
    c.world.positions=lambda:[np.array([-.002])]
    result=motion_planning.execute_request(c,targets,timeout_s=10.,max_actions=80)
    assert result['stop']['reason']=='tracking_progress_stalled'
    assert result['native_actions']==24 and halts==[True]
    assert result['completed_knots']==0 and not result['reached_target']
    assert all(float(command[0][0])==.01 for command in actions)
    assert result['tracking_diagnostics']['roles']['left']['current_max_joint_error_rad']==pytest.approx(.012)
    assert result['tracking_diagnostics']['execution_policy_changed']
    assert 'does not establish' in result['tracking_diagnostics']['guidance']
    assert result['host_adjustments'][-1]['waypoints_skipped']==0
    assert result['host_adjustments'][-1]['target_and_precision']=='unchanged'


def test_slow_but_measurable_joint_progress_is_not_called_stalled(monkeypatch):
    c,targets=controller(monkeypatch,knots=2);actions=[]
    c._action=lambda target,grippers:actions.append(target)
    c.world.positions=lambda:[np.array([-.03+len(actions)*.0002])]
    result=motion_planning.execute_request(c,targets,timeout_s=10.,max_actions=40)
    assert result['native_actions']==40
    assert result['stop']['reason']=='motion_budget_exhausted'
    assert result['completed_knots']==0


def test_guard_window_requires_consistent_out_of_tolerance_measurements():
    from collections import deque
    c=SimpleNamespace(roles=['left'],world=SimpleNamespace(scene=SimpleNamespace(
        snapshot={'robots':[{'arm_joint_names':['joint2']}]})))
    def check(values):
        tracking={'knot_index':1,'actions_at_entry':0,
                  'samples':deque((12+i,[np.array([v])]) for i,v in enumerate(values))}
        return motion_planning.tracking_progress_stop(c,{'left':[]},tracking,24,2)
    assert check([.012]*13)['stalled_roles']==['left']
    assert check([.012]*6+[.007]+[.012]*6) is None
    assert check([.012]*6+[.010]+[.012]*6) is None
    assert check([.012]*12+[float('nan')]) is None


def test_native_or_collision_stop_takes_precedence_over_progress_guard(monkeypatch):
    c,targets=controller(monkeypatch,knots=2);count=[0]
    def action(target,grippers):
        count[0]+=1
        if count[0]==24:c.stopped={'reason':'fixture_collision'}
    c._action=action;c.world.positions=lambda:[np.array([-.002])]
    result=motion_planning.execute_request(c,targets,timeout_s=10.,max_actions=80)
    assert result['stop']['reason']=='fixture_collision'
    assert result['native_actions']==24 and not result['reached_target']
