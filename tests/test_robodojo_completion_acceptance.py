import json
from pathlib import Path
from scripts import robodojo_completion_acceptance as a


def test_all_remaining_task_configs_validate_without_overriding_budgets(tmp_path):
    rows=json.loads(a.INVENTORY.read_text())['tasks']
    remaining=[r for r in rows if r['task'] not in a.PREVIOUSLY_CHECKED]
    assert len(remaining)==40
    for row in remaining:
        out=tmp_path/row['task'];out.mkdir()
        config,env=a.make_config(row,out)
        config.bind_task(env_id=env,task='Official seeded instruction',seed=0)
        components=config.value['components']
        assert components['benchmark']['completion_feedback']==a.PROFILE
        assert 'episode_budget_override' not in components['benchmark']
        if 'Garment' in row['scene_types']:
            assert components['execution']['deformable_mode']=='cloth_visual_short_horizon_v2'
        assert components['benchmark']['env_pattern']==env


def test_unregistered_task_is_not_called_or_reported_as_checker_failure(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'StreamableHttpSimulatorMcpTransport',lambda *args:(_ for _ in ()).throw(AssertionError('No create permitted')))
    row={'task':'unregistered','scene_types':['Fluid'],'native_robot_config':'dual_x5'}
    result=a.run_task(row,tmp_path/'task','unused',set())
    assert result['status']=='blocked_not_registered' and result['checker_status']=='not_run'
    assert result['cleanup']['ok'] and result['native_actions_requested']==0


def test_unknown_creation_outcome_is_not_treated_as_cleaned(tmp_path,monkeypatch):
    from types import SimpleNamespace
    row={'task':'general_pickup','scene_types':['Rigid'],'native_robot_config':'dual_x5'}
    env='openeta/robodojo_dual_x5_general_pickup-v0'
    class Transport:
        def call_tool(self,name,args,timeout_s=None):
            assert name=='create_env'
            raise TimeoutError('lost create response')
    monkeypatch.setattr(a,'StreamableHttpSimulatorMcpTransport',lambda *_:Transport())
    monkeypatch.setattr(a,'make_config',lambda *args:(SimpleNamespace(sha256='fixture'),env))
    result=a.run_task(row,tmp_path/'task','unused',{env})
    assert result['status']=='cleanup_failed'
    assert result['cleanup']['create_outcome_unknown']
    assert not result['cleanup']['ok']


def test_read_only_checks_use_official_text_and_always_close(tmp_path,monkeypatch):
    from types import SimpleNamespace
    row={'task':'general_pickup','scene_types':['Rigid'],'native_robot_config':'dual_x5'}
    env='openeta/robodojo_dual_x5_general_pickup-v0';calls=[]
    class Transport:
        def call_tool(self,name,args,timeout_s=None):
            calls.append(name)
            if name=='create_env':return {'session_id':'owned-session','handle':'owned-env'}
            if name=='reset_env':return {'task':'Official seeded task'}
            if name=='close_env':return {'ok':True}
            raise AssertionError('Unexpected native call')
    monkeypatch.setattr(a,'StreamableHttpSimulatorMcpTransport',lambda *_:Transport())
    config=SimpleNamespace(sha256='fixture',bind_task=lambda **args:args)
    monkeypatch.setattr(a,'make_config',lambda *args:(config,env))
    class Host:
        def __init__(self,args,config,binding,transport):
            assert args.task==binding['task']=='Official seeded task'
            self.transport=transport;self.schemas={'check_task_completion':None}
            self.owned=transport.call_tool('create_env',{'env_id':env,'seed':0,'include_objects':False})
            transport.call_tool('reset_env',{**self.owned,'seed':0})
            self.packet={'action_index':0,'observation':{'objects':[]},'control':{'completion_feedback':a.PROFILE,'episode_budget':{'used_actions':0}}}
        def _status(self):return {'outcome':{'status':'running'}}
        def call(self,name,args):
            assert name=='check_task_completion' and args=={}
            receipt={'status':'checked','profile':a.PROFILE,'feedback_level':'coarse','source_action_index':0,'native_actions':0,'physics_steps':0,'current_conditions_satisfied':None}
            return [SimpleNamespace(text=json.dumps({'receipt':receipt}))]
        def close(self,reason):self.transport.call_tool('close_env',self.owned)
    monkeypatch.setattr(a,'RoboDojoHost',Host)
    result=a.run_task(row,tmp_path/'task','unused',{env})
    assert result['status']=='passed' and result['cleanup']['ok']
    assert calls==['create_env','reset_env','close_env']
    assert result['repeated_receipts_identical'] and result['measured_state_unchanged']
