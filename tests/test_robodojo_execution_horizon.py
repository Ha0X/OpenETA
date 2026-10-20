from types import SimpleNamespace as S
import pytest
from sim.envs.robodojo.execution_horizon import ExecutionHorizon, prepare


def test_rounding_down_never_extends_requested_simulation_time():
    h=ExecutionHorizon(.1,physics_dt=.004,steps_per_action=10)
    assert h.actions==2 and h.maximum_steps==20
    assert h.cap()==2 and h.cap(1)==1 and h.cap(80)==2
    assert ExecutionHorizon(.12,physics_dt=.004,steps_per_action=10).actions==3


@pytest.mark.parametrize('duration',[True,0,-.1,.039,2.001,float('nan'),float('inf')])
def test_invalid_or_subperiod_requests_are_rejected(duration):
    with pytest.raises(ValueError):ExecutionHorizon(duration,physics_dt=.004,steps_per_action=10)


def fake_control():
    c=S(stopped=None,holds=0)
    def halt(reason, *, preserve_gripper_targets=False):
        assert preserve_gripper_targets is True
        c.stopped={'reason':reason};c.holds+=1
    c._halt=halt
    return c


def test_physical_guard_blocks_unexpected_extra_steps_without_another_step():
    from sim.envs.robodojo.checked_control import CheckedController
    h=ExecutionHorizon(.04,physics_dt=.004,steps_per_action=10)
    c=fake_control();c._execution_horizon=h
    for _ in range(10):
        assert h.before_step(c)
        h.after_step()
    advanced=[]
    # Calls the actual production hook: it must return before touching world
    # state or invoking the simulator if an upstream action tries an 11th step.
    CheckedController._guarded_step(c,lambda:advanced.append(True))
    assert not advanced and h.used_steps==10 and c.holds==1
    assert c.stopped['reason']=='requested_execution_horizon_reached'


def test_partial_move_is_reported_as_requested_return_and_holds_without_resume():
    c=fake_control();h=ExecutionHorizon(.1,physics_dt=.004,steps_per_action=10);h.used_steps=20
    r={'physics_steps':20,'reached_target':False,'stop':{'reason':'motion_budget_exhausted','budget_kind':'native_actions'}}
    h.decorate(c,r)
    assert c.holds==1 and r['stop']['reason']=='requested_execution_horizon_reached'
    assert r['stop']['underlying_stop']['reason']=='motion_budget_exhausted'
    assert r['execution_horizon']['actual_duration_s']==.08
    assert not r['execution_horizon']['automatic_resume'] and not r['reached_target']


def test_collision_earlier_limit_and_success_take_precedence():
    for used,reached,stop in [(3,False,{'reason':'measured_collision_after_physics'}),
                             (20,False,{'reason':'measured_collision_after_physics'}),
                             (20,True,None), (10,False,{'reason':'motion_budget_exhausted','budget_kind':'wall_clock'})]:
        c=fake_control();h=ExecutionHorizon(.08,physics_dt=.004,steps_per_action=10);h.used_steps=used
        r={'physics_steps':used,'reached_target':reached,'stop':stop};h.decorate(c,r)
        assert r['stop']==stop and c.holds==0


def test_missing_and_inconsistent_accounting_fail_closed():
    c=fake_control();h=ExecutionHorizon(.04,physics_dt=.004,steps_per_action=10)
    for used,reported in [(1,0),(11,11)]:
        h.used_steps=used
        with pytest.raises(RuntimeError):h.decorate(c,{'physics_steps':reported})


def test_omitted_horizon_preserves_legacy_and_present_horizon_requires_correct_profile():
    c=S(physical_contact=None)
    assert prepare(c,{'operation':'move'}) is None
    with pytest.raises(ValueError):prepare(c,{'operation':'move','max_duration_s':.1})
    c.physical_contact=object();c.native=S(sim=S(physics_dt=.004),obs_manager=S(collect_interval=10))
    assert prepare(c,{'operation':'move','max_duration_s':.1}).actions==2
    with pytest.raises(ValueError):prepare(c,{'operation':'wait','max_duration_s':.1})
    c.native.obs_manager.collect_interval=10.0
    assert prepare(c,{'operation':'move','max_duration_s':.1}).actions==2
    c.native.obs_manager.collect_interval=10.5
    with pytest.raises(ValueError):prepare(c,{'operation':'move','max_duration_s':.1})
