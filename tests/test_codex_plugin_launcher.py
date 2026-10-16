import json
import subprocess
from types import SimpleNamespace

import pytest

from scripts import codex_plugin_smoke as launcher


def fake_login(tmp_path, monkeypatch, *, api_key=False):
    source = tmp_path / "original-codex-home"
    source.mkdir()
    auth = {"auth_mode": "apikey", "OPENAI_API_KEY": "fixture"} if api_key else {
        "auth_mode": "chatgpt", "tokens": {"access_token": "synthetic-test-only"}}
    (source / "auth.json").write_text(json.dumps(auth))
    monkeypatch.setenv("CODEX_HOME", str(source))
    return source


def test_private_install_does_not_modify_login_or_inherit_api_provider(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source / "auth.json").read_bytes()
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-key")
    monkeypatch.setenv("OPENETA_LLM_API_BASE", "https://not-used.invalid")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kw: commands.append((command, kw)))
    home, work, env = launcher.prepare(tmp_path / "run", ["--task", "fixture"], "gpt-5.6-sol", "medium", "codex")
    assert (source / "auth.json").read_bytes() == before
    assert env["CODEX_HOME"] == str(home)
    assert "OPENAI_API_KEY" not in env and "OPENETA_LLM_API_BASE" not in env
    assert home.stat().st_mode & 0o777 == 0o700
    assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
    config = (home / "config.toml").read_text()
    assert 'forced_login_method = "chatgpt"' in config
    assert 'mcp_optional_startup_grace_ms = 0' in config
    assert 'default_tools_approval_mode = "approve"' in config
    assert "shell_tool = false" in config
    assert len(commands) == 2
    assert all(kw["cwd"] == work for _, kw in commands)


def test_api_key_login_is_rejected_before_install(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch, api_key=True)
    with pytest.raises(RuntimeError, match="ChatGPT"):
        launcher.prepare(tmp_path / "run", [], "gpt-5.6-sol", "medium", "codex")
    assert not (tmp_path / "run/codex-home/auth.json").exists()


def test_pinned_catalog_is_private_and_preserves_model_instructions(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch)
    catalog = {"models": [{"slug": "gpt-6-astra", "tool_mode": None,
                          "base_instructions": "unchanged model instructions"}]}
    source = tmp_path / "source-catalog.json"
    source.write_text(json.dumps(catalog))
    before = source.read_bytes()
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: None)
    home, _, _ = launcher.prepare(tmp_path / "run", [], "gpt-6-astra", "medium", "codex",
                                  model_catalog_json=source)
    copied = home / "model-catalog.json"
    assert json.loads(copied.read_text()) == catalog
    assert source.read_bytes() == before
    assert f'model_catalog_json = {json.dumps(str(copied))}' in (home / "config.toml").read_text()
    assert 'shell_tool = false' in (home / "config.toml").read_text()


def test_failed_install_removes_temporary_auth_and_reports_failure(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source / "auth.json").read_bytes()
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1, ["codex", "plugin", "add"])
    monkeypatch.setattr(subprocess, "run", fail)
    output = tmp_path / "run"
    args = SimpleNamespace(output=output, sim_url="http://127.0.0.1:1/sse", env_id="fixture",
        task="fixture", seed=0, timeout=1, max_requests=2, sam3_url="", anygrasp_url="",
        model="gpt-5.6-sol", effort="medium", codex="codex", observe_only=True, controller="mink_joint_velocity")
    assert launcher.run(args) == 1
    assert not (output / "codex-home/auth.json").exists()
    assert (source / "auth.json").read_bytes() == before
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "setup_failed"
    assert not summary["integration_passed"]
    assert not summary["task_success"]


@pytest.mark.parametrize("cleanup_ok,official_success,expected", [
    (True, False, 0), (False, False, 1), (True, True, 0),
])
@pytest.mark.parametrize("with_experience", [False, True])
def test_completed_codex_requires_cleanup_and_separates_task_success(
    tmp_path, monkeypatch, cleanup_ok, official_success, expected, with_experience,
):
    output = tmp_path / "run"
    def prepared(output, *args):
        host_args = args[0]
        assert host_args[host_args.index("--expected-controller") + 1] == "mink_joint_velocity"
        home = output / "codex-home"
        home.mkdir(parents=True)
        (home / "auth.json").write_text("synthetic-test-only")
        (output / "host").mkdir()
        (output / "host/host-status.json").write_text(json.dumps({
            "closed": True, "cleanup": {"ok": cleanup_ok}, "tool_calls": 1,
            "official_task_success": official_success,
        }))
        return home, output, {}
    monkeypatch.setattr(launcher, "prepare", prepared)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw:
        SimpleNamespace(pid=99999999, returncode=0, wait=lambda **kw: 0))
    monkeypatch.setattr(launcher.os, "killpg", lambda *a: None)
    args = SimpleNamespace(output=output, sim_url="http://127.0.0.1:1/sse", env_id="fixture",
        task="fixture", seed=0, timeout=1, max_requests=2, sam3_url="", anygrasp_url="",
        model="gpt-5.6-sol", effort="medium", codex="codex", observe_only=True, controller="mink_joint_velocity")
    if with_experience:
        args.public_experience=experience_fixture(tmp_path)
    assert launcher.run(args) == expected
    summary = json.loads((output / "summary.json").read_text())
    assert summary["integration_passed"] is cleanup_ok
    assert summary["task_success"] is official_success
    assert not (output / "codex-home/auth.json").exists()

    assert summary['evaluation_protocol']==('cross_attempt_public_experience' if with_experience else 'independent_session')
    assert ('A wider contact region stayed stable.' in (output/'prompt.txt').read_text()) is with_experience
    assert (output/'public-experience.json').exists() is with_experience


def test_native_route_deadline_does_not_preempt_the_host_episode(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    output = tmp_path / 'run'
    launcher.prepare(output, ['--timeout', '2400'], 'gpt-6-astra', 'high', 'codex')
    config = json.loads((output / 'marketplace/plugins/openeta/.mcp.json').read_text())
    assert config['openeta']['tool_timeout_sec'] == 2430
    assert config['openeta']['startup_timeout_sec'] == 180


def test_robodojo_startup_covers_create_reset_and_first_observation(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: None)
    output = tmp_path / 'run'
    launcher.prepare(output, ['--expected-controller', 'robodojo_checked', '--timeout', '4800'],
                     'gpt-6-astra', 'high', 'codex')
    config = json.loads((output / 'marketplace/plugins/openeta/.mcp.json').read_text())['openeta']
    assert config['startup_timeout_sec'] == 240 + 180 + 240 + 60
    assert config['tool_timeout_sec'] == 4830


def test_launcher_closes_only_matching_recorded_owner(tmp_path, monkeypatch):
    from tools import harness_transport
    args = SimpleNamespace(controller='robodojo_checked', sim_url='http://example.invalid/mcp', env_id='fixture')
    (tmp_path / 'host').mkdir()
    path = tmp_path / 'host/owned-environment.json'
    owned = dict(sim_url=args.sim_url, env_id=args.env_id, session_id='session', handle='handle')
    calls = []
    class Transport:
        def __init__(self, url):
            assert url == args.sim_url
        def call_tool(self, name, arguments, **kwargs):
            calls.append((name, arguments, kwargs))
            return {'ok': True}
    monkeypatch.setattr(harness_transport, 'StreamableHttpSimulatorMcpTransport', Transport)
    for key in ('sim_url', 'env_id', 'session_id', 'handle'):
        path.write_text(json.dumps({**owned, key: ''}))
        assert not launcher.cleanup_recorded_robodojo_environment(tmp_path, args, {})['ok']
        assert not calls
    path.write_text(json.dumps(owned))
    assert launcher.cleanup_recorded_robodojo_environment(tmp_path, args, {'closed': True, 'cleanup': {'ok': True}}) is None
    assert not calls
    assert launcher.cleanup_recorded_robodojo_environment(tmp_path, args, {})['ok']
    assert calls == [('close_env', {'session_id': 'session', 'handle': 'handle'}, {'timeout_s': 90})]


def test_startup_timeout_is_not_a_completed_robot_trial(tmp_path, monkeypatch):
    output = tmp_path / 'run'
    def prepared(output, *args):
        home = output / 'codex-home'
        home.mkdir(parents=True)
        (home / 'auth.json').write_text('synthetic-test-only')
        return home, output, {}
    def spawn(*args, **kwargs):
        kwargs['stderr'].write('MCP server startup failed server_name="openeta": timed out handshaking\n')
        return SimpleNamespace(pid=99999999, returncode=0, wait=lambda **kw: 0)
    monkeypatch.setattr(launcher, 'prepare', prepared)
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    monkeypatch.setattr(launcher.os, 'killpg', lambda *a: None)
    args = SimpleNamespace(output=output, sim_url='http://example.invalid/mcp', env_id='fixture',
        task='fixture', seed=0, timeout=1, max_requests=2, sam3_url='', anygrasp_url='',
        model='gpt-6-astra', effort='high', codex='codex', observe_only=True, controller='robodojo_checked')
    assert launcher.run(args) == 1
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['status'] == 'host_startup_failed'
    assert summary['host_startup_error'] == {'code': 'mcp_startup_timeout', 'robot_tool_calls_observed': 0}
    assert not summary['integration_passed'] and not summary['task_success']
    assert not (output / 'codex-home/auth.json').exists()


def test_https_transport_keeps_subscription_auth_and_default_endpoint(tmp_path, monkeypatch):
    import tomllib
    source=fake_login(tmp_path,monkeypatch)
    before=(source/'auth.json').read_bytes()
    monkeypatch.setenv('OPENAI_API_KEY','synthetic-key')
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    home,_,env=launcher.prepare(tmp_path/'run',[],'gpt-6-astra','high','codex',model_transport='https')
    config=tomllib.loads((home/'config.toml').read_text())
    provider=config['model_providers'][config['model_provider']]
    assert config['forced_login_method']=='chatgpt'
    assert provider['requires_openai_auth'] is True and provider['supports_websockets'] is False
    assert provider['wire_api']=='responses'
    assert provider['name']=='OpenAI'  # CLI requires canonical name for zstd request compression.
    assert all(key not in provider for key in ('base_url','env_key','experimental_bearer_token'))
    assert 'OPENAI_API_KEY' not in env
    assert (source/'auth.json').read_bytes()==before


def test_operator_interruption_keeps_outcome_and_cleans_private_auth(tmp_path,monkeypatch):
    output=tmp_path/'run'
    def prepared(output,*args):
        home=output/'codex-home';home.mkdir(parents=True)
        (home/'auth.json').write_text('synthetic-test-only')
        (output/'host').mkdir()
        (output/'host/host-status.json').write_text(json.dumps({'closed':True,'cleanup':{'ok':True},
            'tool_calls':1,'official_task_success':False,'outcome':{'status':'running','terminal':False}}))
        return home,output,{}
    calls=[]
    def wait(**kw):
        calls.append(1)
        if len(calls)==1:raise KeyboardInterrupt
        return -15
    monkeypatch.setattr(launcher,'prepare',prepared)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999999,returncode=-15,wait=wait))
    killed=[];monkeypatch.setattr(launcher.os,'killpg',lambda *a:killed.append(a))
    args=SimpleNamespace(output=output,sim_url='http://127.0.0.1:1/sse',env_id='fixture',
        task='fixture',seed=0,timeout=1,max_requests=2,sam3_url='',anygrasp_url='',
        model='gpt-6-astra',effort='high',codex='codex',observe_only=True,controller='mink_joint_velocity')
    assert launcher.run(args)==1
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='operator_interrupted'
    assert summary['host']['outcome']['status']=='running'
    assert not summary['task_success'] and not summary['integration_passed']
    assert killed and not (output/'codex-home/auth.json').exists()


def experience_fixture(tmp_path, *, event_type='agent_message', completed=True, text='A wider contact region stayed stable.'):
    import hashlib
    source=tmp_path/'prior-events.jsonl'
    lines=[{'type':'item.completed','item':{'type':event_type,'text':text}}]
    if completed:lines.append({'type':'turn.completed'})
    source.write_text(''.join(json.dumps(x)+'\n' for x in lines))
    path=tmp_path/'selection.json'
    path.write_text(json.dumps({'schema':'openeta.public_episode_experience.v1','entries':[
        {'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'event_index':0}]}))
    return path


def test_public_experience_is_exact_public_quote_with_protocol_label(tmp_path):
    path=experience_fixture(tmp_path)
    prompt,record=launcher.public_experience(path)
    assert record['quoted_messages']==['A wider contact region stayed stable.']
    assert record['independent_pass_at_k'] is False
    assert 'Reobserve the new episode' in prompt
    assert str(tmp_path) not in prompt


@pytest.mark.parametrize('kind',['tool_payload','incomplete_run','modified_source','missing_event','negative_index','oversized'])
def test_public_experience_rejects_nonpublic_or_unverifiable_sources(tmp_path,kind):
    path=experience_fixture(tmp_path,event_type='mcp_tool_call' if kind=='tool_payload' else 'agent_message',
                            completed=kind!='incomplete_run',text='x'*6001 if kind=='oversized' else 'public text')
    if kind=='modified_source':
        with (tmp_path/'prior-events.jsonl').open('a') as f:f.write('\n')
    if kind in ('missing_event','negative_index'):
        value=json.loads(path.read_text());value['entries'][0]['event_index']=10 if kind=='missing_event' else -1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):launcher.public_experience(path)



def test_robodojo_packages_readonly_guides_and_keeps_libero_separate(tmp_path, monkeypatch):
    from tools.robodojo_skill_library import BUNDLES, PREFIX, SkillLibrary
    from tools.harness_config import ConfigRegistry
    fake_login(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: None)
    output=tmp_path/'robodojo'
    home,_,_=launcher.prepare(output,['--expected-controller','robodojo_checked'],
        'gpt-6-astra','high','codex')
    mcp=json.loads((output/'marketplace/plugins/openeta/.mcp.json').read_text())['openeta']
    root=output/'marketplace'/PREFIX
    assert mcp['env']['OPENETA_ROBODOJO_SKILL_ROOT']==str(root)
    assert 'shell_tool = false' in (home/'config.toml').read_text()
    config=ConfigRegistry().resolve('robodojo_dual_x5_checked_v1')
    library=SkillLibrary(config.value['components']['interface']['source_sha256'],root=root)
    for skill_id in BUNDLES:assert library.read(skill_id)['documents']
    assert not (output/'marketplace/plugins/openeta/skills/openeta-pick').exists()
    other=tmp_path/'libero'
    launcher.prepare(other,['--expected-controller','mink_joint_velocity','--tool-profile','atomic'],
        'gpt-6-astra','high','codex')
    config=json.loads((other/'marketplace/plugins/openeta/.mcp.json').read_text())['openeta']
    assert 'OPENETA_ROBODOJO_SKILL_ROOT' not in config['env']
    assert not (other/'marketplace'/PREFIX).exists()


@pytest.mark.parametrize('config_id', ['robodojo_dual_x5_make_kong_auxiliary_v103', 'robodojo_dual_x5_press_by_number_repair_v109', 'robodojo_dual_x5_organize_table_repair_v109'])
def test_kong_supplement_reaches_atomic_model_prompt_and_experiment_record(tmp_path, monkeypatch, config_id):
    from tools.harness_config import ConfigRegistry
    from tools import harness_binding
    from tools.task_rule_supplements import KONG_ENV, KONG_RULE
    config=ConfigRegistry().resolve(config_id)
    from tools.task_rule_supplements import task_rule_supplement
    benchmark = config.value['components']['benchmark']
    KONG_ENV = benchmark['env_pattern']
    _, expected_rule = task_rule_supplement(benchmark, KONG_ENV)
    KONG_RULE = expected_rule['text']
    monkeypatch.setattr(harness_binding,'preflight',lambda args:(config,{}))
    output=tmp_path/'rule-run'
    def prepared(output,*args):
        home=output/'codex-home';home.mkdir(parents=True)
        (home/'auth.json').write_text('synthetic-test-only')
        (output/'host').mkdir()
        (output/'host/host-status.json').write_text(json.dumps(dict(closed=True,cleanup={'ok':True},tool_calls=1,official_task_success=False)))
        skill=output/'marketplace/plugins/openeta/skills/openeta-robodojo/SKILL.md'
        skill.parent.mkdir(parents=True);skill.write_text('---\nname: fixture\n---\nUse public robot tools.')
        return home,output,{}
    monkeypatch.setattr(launcher,'prepare',prepared)
    commands=[]
    def spawn(command,**kwargs):
        commands.append(command)
        return SimpleNamespace(pid=99999999,returncode=0,wait=lambda **kw:0)
    monkeypatch.setattr(subprocess,'Popen',spawn)
    monkeypatch.setattr(launcher.os,'killpg',lambda *a:None)
    args=SimpleNamespace(output=output,sim_url='http://example.invalid/mcp',env_id=KONG_ENV,
        task='Wait for the opponent to discard a tile, then declare a kong with the matching tiles.',
        seed=0,timeout=1,max_requests=2,sam3_url='',anygrasp_url='',model='gpt-6-astra',effort='high',
        codex='codex',observe_only=False,controller='robodojo_checked',tool_profile='atomic',
        harness_config=config.id,harness_config_sha256=config.sha256)
    assert launcher.run(args)==0
    prompt=(output/'prompt.txt').read_text()
    assert prompt.startswith(args.task) and prompt.count(KONG_RULE)==1
    assert KONG_RULE in commands[0][-1]
    record=json.loads((output/'task-rule-supplement.json').read_text())
    summary=json.loads((output/'summary.json').read_text())
    assert summary['experiment_factors']['task_rule_supplement']==record
    assert record['comparable_to_original_instruction'] is False
    assert not (output/'codex-home/auth.json').exists()


def test_no_episode_deadline_keeps_finite_mcp_timeout(tmp_path,monkeypatch):
    fake_login(tmp_path,monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    out=tmp_path/'run'
    launcher.prepare(out,['--expected-controller','robodojo_checked','--timeout','0'],
                     'gpt-6-astra','high','codex')
    config=json.loads((out/'marketplace/plugins/openeta/.mcp.json').read_text())['openeta']
    assert config['tool_timeout_sec']==480 and config['startup_timeout_sec']==720


def test_no_episode_deadline_launcher_waits_without_timeout_and_records_elapsed(tmp_path,monkeypatch):
    waits=[];out=tmp_path/'run'
    def prepared(output,*args):
        home=output/'codex-home';home.mkdir(parents=True)
        (home/'auth.json').write_text('synthetic-test-only')
        (output/'host').mkdir()
        (output/'host/host-status.json').write_text(json.dumps({'closed':True,'cleanup':{'ok':True},'tool_calls':1,'official_task_success':False}))
        return home,output,{}
    monkeypatch.setattr(launcher,'prepare',prepared)
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw:SimpleNamespace(pid=99999999,returncode=0,wait=lambda **kw:waits.append(kw.get('timeout')) or 0))
    monkeypatch.setattr(launcher.os,'killpg',lambda *a:None)
    args=SimpleNamespace(output=out,sim_url='http://127.0.0.1:1/mcp',env_id='fixture',task='fixture',
        seed=0,timeout=0,max_requests=200,sam3_url='',anygrasp_url='',model='gpt-6-astra',
        effort='high',codex='codex',observe_only=True,controller='robodojo_checked')
    assert launcher.run(args)==0 and waits[0] is None
    summary=json.loads((out/'summary.json').read_text())
    assert summary['limits']['episode_timeout_s'] is None
    assert summary['limits']['max_requests']==200 and summary['elapsed_s']>=0
