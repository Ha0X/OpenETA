from types import SimpleNamespace as S
import pytest
from sim.envs.robodojo.initialization_profile import ENV, PROFILE, prepare_reset, inspect_binding


def fixture(monkeypatch, verdict=True, task='fill_egg_holder'):
    monkeypatch.setenv(ENV, PROFILE)
    events=[]
    def check(env):
        events.append('check')
        env.success=verdict
        return verdict, [] if verdict else [0]
    native=S(scene_manager=S(layout_manager=S(check_layout_stability=check)),
             sim_step=lambda **kw:events.append('step'), success=True)
    adapter=S(_env=native,_task_name=task,_seed=0)
    return adapter, events


def test_wait_precedes_native_check_and_repeated_reset_does_not_stack(monkeypatch):
    a, events=fixture(monkeypatch)
    for seed in (0,1):
        a._seed=seed
        prepare_reset(a)
        with pytest.raises(ValueError,match='did not pass'):inspect_binding(a,PROFILE)
        result=a._env.scene_manager.layout_manager.check_layout_stability(a._env)
        assert result == (True, [])
        receipt=inspect_binding(a,PROFILE)
        assert receipt['seed']==seed and receipt['completed_steps']==300
        assert receipt['native_stability_passed'] is True
        assert receipt['comparable_to_native_initialization_baseline'] is False
    assert events==(['step']*300+['check'])*2
    with pytest.raises(ValueError,match='differs'):inspect_binding(a,None)


def test_native_rejection_remains_rejection(monkeypatch):
    a,_=fixture(monkeypatch,verdict=False);prepare_reset(a)
    assert a._env.scene_manager.layout_manager.check_layout_stability(a._env)==(False,[0])
    assert a._env.success is False
    with pytest.raises(ValueError,match='did not pass'):inspect_binding(a,PROFILE)


def test_partial_wait_exception_cannot_admit_or_run_native_check(monkeypatch):
    a,events=fixture(monkeypatch)
    def step(**kw):
        if len(events)==7:raise RuntimeError('physics failed')
        events.append('step')
    a._env.sim_step=step;prepare_reset(a)
    with pytest.raises(RuntimeError,match='physics failed'):
        a._env.scene_manager.layout_manager.check_layout_stability(a._env)
    assert events==['step']*7
    assert a._env._openeta_initialization_receipt['completed_steps']==7
    with pytest.raises(ValueError):inspect_binding(a,PROFILE)


def test_other_task_and_native_default_unchanged(monkeypatch):
    for task,enabled in [('stack_blocks',True),('fill_egg_holder',False)]:
        a,events=fixture(monkeypatch,task=task)
        if not enabled:monkeypatch.delenv(ENV)
        original=a._env.scene_manager.layout_manager.check_layout_stability
        prepare_reset(a)
        assert a._env.scene_manager.layout_manager.check_layout_stability is original
        assert inspect_binding(a,None) is None
        with pytest.raises(ValueError):inspect_binding(a,PROFILE)
        assert events==[]


def test_profile_drift_cannot_silently_become_baseline(monkeypatch):
    a,_=fixture(monkeypatch);prepare_reset(a)
    a._env.scene_manager.layout_manager.check_layout_stability(a._env)
    monkeypatch.delenv(ENV)
    with pytest.raises(ValueError):inspect_binding(a,None)
    prepare_reset(a)
    assert inspect_binding(a,None) is None


def test_config_is_exactly_task_scoped():
    from tools.harness_config import ConfigRegistry, ConfigError, _validate_component
    c=ConfigRegistry().resolve('robodojo_dual_x5_egg_settle_v107')
    c.require_executable()
    b=c.value['components']['benchmark']
    assert b['initialization_profile']==PROFILE
    b['env_pattern']='openeta/robodojo_dual_x5_stack_blocks-v0'
    with pytest.raises(ConfigError) as exc:
        _validate_component('benchmark',b['id'],b)
    assert exc.value.code == 'invalid_initialization_profile'
