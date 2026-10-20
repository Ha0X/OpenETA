from types import SimpleNamespace
import pytest
from sim.envs.robodojo.episode_protocol import EXTENDED_ALIGN, bind_protocol, reset_protocol
from sim.envs.robodojo.worker_bridge import episode_budget


def adapter():
    return SimpleNamespace(_env=SimpleNamespace(step_lim=200,take_action_cnt=[0]),
        _task_name='align_blocks',_robot_variant='dual_x5',_step_count=0,
        _terminal=False,_checked_controller=None,_max_episode_steps=1000)


def test_explicit_profile_changes_quota_only_and_reset_restores_native():
    a=adapter();a._env.success=[True];a._env.end_flag=[False]
    bind_protocol(a,EXTENDED_ALIGN,'status')
    assert a._env.step_lim==400 and a._env.success==[True] and a._env.end_flag==[False]
    a._step_count=205;a._env.take_action_cnt=[205]
    bind_protocol(a,EXTENDED_ALIGN,'move')
    report=episode_budget(a)
    assert report['remaining_actions']==195 and report['action_limit']==400
    assert report['protocol']['comparable_to_native_budget_baseline'] is False
    with pytest.raises(ValueError,match='within an episode'):bind_protocol(a,None,'status')
    reset_protocol(a)
    assert a._env.step_lim==200 and a._episode_protocol is None
    bind_protocol(a,None,'status')
    assert a._env.step_lim==200


def test_default_untouched_and_cannot_promote_after_binding():
    a=adapter();bind_protocol(a,None,'status')
    assert a._env.step_lim==200
    with pytest.raises(ValueError):bind_protocol(a,EXTENDED_ALIGN,'status')
    assert a._env.step_lim==200


@pytest.mark.parametrize('field,value',[('_task_name','cover_blocks'),('_robot_variant','single_x5'),
    ('_step_count',1),('_terminal',True),('_checked_controller',object()),('_max_episode_steps',300)])
def test_rejects_wrong_task_or_late_override(field,value):
    a=adapter();setattr(a,field,value)
    with pytest.raises(ValueError):bind_protocol(a,EXTENDED_ALIGN,'status')
    assert a._env.step_lim==200


@pytest.mark.parametrize('limit,counts',[(201,[0]),(200,[1]),(200,[0,0]),(None,[0])])
def test_rejects_native_protocol_drift(limit,counts):
    a=adapter();a._env.step_lim=limit;a._env.take_action_cnt=counts
    with pytest.raises(ValueError):bind_protocol(a,EXTENDED_ALIGN,'status')
    assert a._env.step_lim==limit


def test_limit_drift_and_non_status_override_rejected():
    a=adapter()
    with pytest.raises(ValueError):bind_protocol(a,EXTENDED_ALIGN,'move')
    bind_protocol(a,EXTENDED_ALIGN,'status');a._env.step_lim=500
    with pytest.raises(ValueError,match='drifted'):bind_protocol(a,EXTENDED_ALIGN,'status')


def tower_adapter():
    a=adapter();a._task_name='build_tower';a._env.step_lim=1050
    return a


@pytest.mark.parametrize('profile,limit',[('build_tower_2000_v1',2000),('build_tower_4000_v1',4000)])
def test_tower_budget_binds_both_layers_and_restores(profile,limit):
    EXTENDED_TOWER=profile
    a=tower_adapter();a._env.success=[False]
    bind_protocol(a,EXTENDED_TOWER,'status')
    assert a._env.step_lim==a._max_episode_steps==limit
    assert a._env.success==[False]
    for used in (1000,1500,limit-1):
        a._step_count=used;a._env.take_action_cnt=[used]
        bind_protocol(a,EXTENDED_TOWER,'move')
        assert episode_budget(a)['remaining_actions']==limit-used
    reset_protocol(a)
    assert a._env.step_lim==1050 and a._max_episode_steps==1000


@pytest.mark.parametrize('layer',['native','adapter'])
def test_tower_rejects_wrong_original_and_bound_limit_drift(layer):
    from sim.envs.robodojo.episode_protocol import EXTENDED_TOWER
    a=tower_adapter()
    if layer=='native':a._env.step_lim=1051
    else:a._max_episode_steps=1001
    before=(a._env.step_lim,a._max_episode_steps)
    with pytest.raises(ValueError):bind_protocol(a,EXTENDED_TOWER,'status')
    assert (a._env.step_lim,a._max_episode_steps)==before
    a=tower_adapter();bind_protocol(a,EXTENDED_TOWER,'status')
    if layer=='native':a._env.step_lim=1000
    else:a._max_episode_steps=1000
    with pytest.raises(ValueError,match='drifted'):bind_protocol(a,EXTENDED_TOWER,'move')


@pytest.mark.parametrize('task,limit',[('play_tic_tac_toe',1100),('imitate_sorting_sequence',1600)])
def test_opt_in_adapter_alignment_preserves_native_budget_and_outcome(task,limit):
    a=adapter();a._task_name=task;a._env.step_lim=limit
    a._env.success=[False];a._env.end_flag=[False]
    profile=task+'_native_adapter_v101'
    bind_protocol(a,profile,'status')
    assert a._env.step_lim==limit and a._max_episode_steps==limit
    assert a._env.success==[False] and a._env.end_flag==[False]
    a._step_count=1000;a._env.take_action_cnt=[1000]
    report=episode_budget(a)
    assert report['remaining_actions']==limit-1000
    assert report['protocol']['evaluation_variant']=='native_limit_adapter_alignment'
    # The original harness cap changed, so do not pool it with that baseline.
    assert report['protocol']['comparable_to_native_budget_baseline'] is False
    with pytest.raises(ValueError):bind_protocol(a,None,'status')
    reset_protocol(a)
    assert a._env.step_lim==limit and a._max_episode_steps==1000


@pytest.mark.parametrize('field,value',[('_task_name','make_kong'),('_step_count',1),('_max_episode_steps',900)])
def test_adapter_alignment_rejects_wrong_task_late_binding_or_explicit_other_quota(field,value):
    a=adapter();a._task_name='play_tic_tac_toe';a._env.step_lim=1100
    setattr(a,field,value)
    with pytest.raises(ValueError):bind_protocol(a,'play_tic_tac_toe_native_adapter_v101','status')
    assert a._env.step_lim==1100 and a._max_episode_steps==(900 if field=='_max_episode_steps' else 1000)


def test_tic_tac_toe_extended_budget_crosses_original_cap_and_restores():
    a=adapter();a._task_name='play_tic_tac_toe';a._env.step_lim=1100
    a._env.success=[False];a._env.end_flag=[False]
    bind_protocol(a,'play_tic_tac_toe_1300_v104','status')
    assert a._env.step_lim==a._max_episode_steps==1300
    a._step_count=1101;a._env.take_action_cnt=[1101]
    report=episode_budget(a)
    assert report['remaining_actions']==199
    assert report['protocol']['evaluation_variant']=='extended_action_budget'
    assert report['protocol']['comparable_to_native_budget_baseline'] is False
    assert a._env.success==[False] and a._env.end_flag==[False]
    reset_protocol(a)
    assert a._env.step_lim==1100 and a._max_episode_steps==1000


@pytest.mark.parametrize('variant',['single_x5','dual_x5'])
def test_unlimited_budget_counts_past_both_caps_and_serializes_without_infinity(variant):
    import json
    from sim.envs.robodojo.episode_protocol import UNLIMITED_ACTIONS
    a=adapter();a._robot_variant=variant;a._env.success=[True];a._env.end_flag=[False]
    bind_protocol(a,UNLIMITED_ACTIONS,'status')
    a._step_count=5000;a._env.take_action_cnt=[5001]
    bind_protocol(a,UNLIMITED_ACTIONS,'move')
    report=episode_budget(a)
    assert report['unlimited'] and report['action_limit'] is None and report['remaining_actions'] is None
    assert [x['used_actions'] for x in report['limits']]==[5000,5001]
    assert report['protocol']['original_native_action_limit']==200
    assert a._step_count < a._max_episode_steps and a._env.take_action_cnt[0] < a._env.step_lim
    assert a._env.success==[True] and a._env.end_flag==[False]
    json.dumps(report,allow_nan=False)
    a._terminal=True
    assert episode_budget(a)['terminal'] is True
    reset_protocol(a)
    assert a._env.step_lim==200 and a._max_episode_steps==1000


@pytest.mark.parametrize('layer',['native','adapter'])
def test_unlimited_budget_rejects_reintroduced_cap(layer):
    from sim.envs.robodojo.episode_protocol import UNLIMITED_ACTIONS
    a=adapter();bind_protocol(a,UNLIMITED_ACTIONS,'status')
    if layer=='native':a._env.step_lim=1000
    else:a._max_episode_steps=1000
    with pytest.raises(ValueError,match='drifted'):episode_budget(a)


@pytest.mark.parametrize('field,value',[('_step_count',1),('_terminal',True),('_checked_controller',object())])
def test_unlimited_budget_cannot_be_enabled_after_control_begins(field,value):
    from sim.envs.robodojo.episode_protocol import UNLIMITED_ACTIONS
    a=adapter();setattr(a,field,value)
    with pytest.raises(ValueError):bind_protocol(a,UNLIMITED_ACTIONS,'status')
    assert a._env.step_lim==200 and a._max_episode_steps==1000
