from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.envs.robodojo.bounded_wait import wait


def control(limit=20, stop=False, terminal_after=None):
    adapter=S(_max_episode_steps=limit, _step_count=0, _terminal=False, _env=None)
    c=S(adapter=adapter, physical_contact=object(), roles=['left','right'],
        native=S(sim=S(physics_dt=.01), obs_manager=S(collect_interval=4)),
        world=S(refresh=lambda:None, positions=lambda:[np.array([.1,.2]),np.array([.3,.4])],
            scene=S(snapshot={'robots':[{'joint_names':['a','g'],'arm_joint_names':['a']}]*2})),
        _grippers=lambda:[0.,1.], calls=[])
    def action(targets, grips):
        c.calls.append(([x.copy() for x in targets],list(grips)))
        adapter._step_count += 1
        c.physics_steps += 1 if stop else 4
        if stop:c.stopped={'reason':'collision'}
        if terminal_after and adapter._step_count >= terminal_after:adapter._terminal=True
    c._action=action
    return c


def test_wait_billed_quantized_and_holds_both_arms_and_grippers():
    c=control(); r=wait(c,.1)
    assert r['native_actions']==c.adapter._step_count==3
    assert r['actual_duration_s']==.12 and r['status']=='completed'
    assert r['hold_arm_joint_positions']=={'left':[.1],'right':[.3]}
    assert all(g==[0.,1.] for _,g in c.calls)
    assert all(np.array_equal(t[0],[.1]) for t,_ in c.calls)


@pytest.mark.parametrize('duration',[0,-1,2.01,float('nan'),float('inf'),True])
def test_invalid_wait_does_not_execute(duration):
    c=control()
    with pytest.raises(ValueError):wait(c,duration)
    assert not c.calls


def test_wait_stops_mid_interval_and_reports_actual_time():
    c=control(stop=True);r=wait(c,1.)
    assert r['native_actions']==1 and r['physics_steps']==1
    assert r['actual_duration_s']==.01 and r['status']=='interrupted'
    assert r['stop']['reason']=='collision'


@pytest.mark.parametrize('kwargs',[{'limit':1},{'terminal_after':1}])
def test_wait_respects_native_terminal_and_budget(kwargs):
    c=control(**kwargs);r=wait(c,1.)
    assert r['native_actions']==1 and r['actual_duration_s']==.04
    assert r['stop']['reason']=='episode_terminal_or_budget_exhausted'


def test_unlimited_episode_wait_still_bounded_and_stops_on_native_terminal():
    from sim.envs.robodojo.episode_protocol import bind_protocol, UNLIMITED_ACTIONS
    c=control(1000,terminal_after=1603)
    c.adapter._env=S(step_lim=1600,take_action_cnt=[0])
    c.adapter._task_name='imitate_sorting_sequence';c.adapter._robot_variant='dual_x5'
    bind_protocol(c.adapter,UNLIMITED_ACTIONS,'status')
    c.adapter._step_count=1600;c.adapter._env.take_action_cnt=[1600]
    result=wait(c,1.)
    assert result['native_actions']==3
    assert c.adapter._step_count==1603
    assert result['stop']['reason']=='episode_terminal_or_budget_exhausted'
