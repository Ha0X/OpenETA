from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.envs.robodojo.motion_precision import tolerance, decorate
from sim.envs.robodojo import motion_planning
from test_robodojo_terminal_tracking import controller


@pytest.mark.parametrize('value',[True,0,.00049,.00301,float('nan'),float('inf'),'0.001'])
def test_bad_precision_rejected(value):
    with pytest.raises(ValueError):tolerance(value)


def test_explicit_precision_does_not_silently_revert_to_3mm(monkeypatch):
    c,targets=controller(monkeypatch);c.position_tolerance_m=.0005
    r=motion_planning.execute_request(c,targets,timeout_s=10.,max_actions=4)
    assert not r['reached_target'] and r['native_actions']==4
    assert .0005 < r['final_tcp_error']['left']['position_m'] < .003
    assert r['host_adjustments'][0]['position_tolerance_m']==.0005


def test_requested_precision_can_complete_without_goal_change(monkeypatch):
    c,targets=controller(monkeypatch);c.position_tolerance_m=.0015
    original=np.array(targets['left'][0],copy=True)
    r=motion_planning.execute_request(c,targets,timeout_s=10.,max_actions=4)
    assert r['reached_target'] and r['native_actions']==2
    assert np.array_equal(original,targets['left'][0])


def test_final_criterion_covers_legacy_recovery_receipts():
    actual=np.eye(4);actual[0,3]=.001
    c=S(roles=['left'],tcp=lambda i:actual)
    r=decorate({'reached_target':True},c,{'left':[np.eye(4)]},.0005)
    assert not r['reached_target'] and r['stop']['reason']=='measured_final_pose_outside_requested_tolerance'
    assert not r['completion_criteria']['target_modified']


def test_precision_failure_cannot_keep_completed_status():
    actual=np.eye(4);actual[0,3]=.001
    c=S(roles=['left'],tcp=lambda i:actual)
    r=decorate({'reached_target':True,'status':'completed','native_actions':3},c,{'left':[np.eye(4)]},.0005)
    assert r['status']=='interrupted' and not r['reached_target']
