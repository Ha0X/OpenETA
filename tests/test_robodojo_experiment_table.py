from scripts.update_robodojo_experiment_table import classify


def test_completed_model_text_does_not_prove_task_success():
    assert classify({'status':'completed','task_success':True},
                    {'tool_calls':5,'outcome':{'status':'running'}}) == (1,0,'模型结束但任务未成功')


def test_only_verified_native_terminal_success_counts():
    host={'tool_calls':5,'outcome':{'status':'success','benchmark_success':True,
          'terminal':True,'source':'robodojo.eval_env.terminal_flags'}}
    assert classify({'status':'completed'},host) == (1,1,'官方成功')
    host['outcome']['source']='model_claim'
    assert classify({'status':'completed'},host)[1] == 0


def test_startup_failure_excluded_but_started_transport_failure_retained():
    assert classify({'status':'codex_failed'},{'tool_calls':0}) == (0,0,'未进入任务控制')
    assert classify({'status':'codex_failed'},{'tool_calls':9}) == (1,0,'模型或传输异常')
    assert classify({'status':'operator_interrupted'},{'tool_calls':9}) == (1,0,'人工中断')


def test_one_shot_and_zero_shot_separate_columns_and_native_quota(tmp_path,monkeypatch):
    import json
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    columns=[]
    for condition in ('zero_shot','one_shot_task_specific'):
        trial=tmp_path/('model-v75-'+condition);run=trial/'run';run.mkdir(parents=True)
        (trial/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_build_tower-v0','seed':0}))
        summary={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed',
            'experiment_factors':{'conditioning':condition},
            'host':{'tool_calls':10,'simulation_budget':{'source':'openeta_adapter_max_episode_steps','action_limit':1000,
                'limits':[{'source':'robodojo_native_step_lim','action_limit':1050}]},'outcome':{'status':'running'}}}
        p=run/'summary.json';p.write_text(json.dumps(summary))
        entry=table.record(p,summary,{'build_tower'})
        columns.append(entry['column']);assert entry['native_action_limit']==1050
    assert columns==['v75 · 零示范','v75 · 任务示范1-shot']


def test_completion_oracle_has_explicit_separate_column(tmp_path,monkeypatch):
    import json
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    trial=tmp_path/'model-v78-one_shot_task_specific';run=trial/'run';run.mkdir(parents=True)
    (trial/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_build_tower-v0'}))
    s={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed',
       'experiment_factors':{'conditioning':'one_shot_task_specific','completion_feedback':'build_tower_semantic_v1'},
       'host':{'tool_calls':10,'completion_checks':2,'simulation_budget':{'protocol':{'evaluation_variant':'extended_action_budget','effective_native_action_limit':4000}},'outcome':{'status':'running'}}}
    path=run/'summary.json';path.write_text(json.dumps(s));entry=table.record(path,s,{'build_tower'})
    assert entry['column']=='v78 · 扩额4000 · 任务示范1-shot · 判别器反馈'
    assert entry['completion_checks']==2


def test_explicit_version_records_arbitrarily_named_run_without_guessing(tmp_path,monkeypatch):
    import json
    import pytest
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    run=tmp_path/'astra-high-attempt1';run.mkdir()
    experiment={'env_id':'openeta/robodojo_dual_x5_pick_from_conveyor_by_image-v0','harness_version':'v85'}
    exp=run/'experiment.json';exp.write_text(json.dumps(experiment))
    s={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed','host':{'tool_calls':10,'outcome':{'status':'running'}}}
    path=run/'summary.json';path.write_text(json.dumps(s))
    entry=table.record(path,s,{'pick_from_conveyor_by_image'})
    assert entry['harness_version']=='v85' and entry['version_source']=='astra-high-attempt1/experiment.json'
    experiment['harness_version']='unknown';exp.write_text(json.dumps(experiment))
    with pytest.raises(ValueError,match='Invalid explicit'):table.record(path,s,{'pick_from_conveyor_by_image'})
    other=tmp_path/'model-v84';other.mkdir();experiment['harness_version']='v85'
    (other/'experiment.json').write_text(json.dumps(experiment));path=other/'summary.json';path.write_text(json.dumps(s))
    with pytest.raises(ValueError,match='Conflicting'):table.record(path,s,{'pick_from_conveyor_by_image'})


def test_deployment_correction_is_bound_to_original_harness_record(tmp_path, monkeypatch):
    import json
    import hashlib
    import pytest
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    run=tmp_path/'model-v90';(run/'host').mkdir(parents=True)
    exp=run/'experiment.json';exp.write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_build_tower-v0','release':'old-label'}))
    original=exp.read_bytes()
    binding=run/'host/harness-binding.json';binding.write_text('{}')
    correction=run/'deployment-verification.json'
    correction.write_text(json.dumps({'schema':'openeta.deployment_verification.v1',
        'verified_release':'verified-candidate','harness_binding_sha256':hashlib.sha256(binding.read_bytes()).hexdigest()}))
    summary={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed',
             'host':{'tool_calls':3,'outcome':{'status':'running'}}}
    path=run/'summary.json';path.write_text(json.dumps(summary))
    entry=table.record(path,summary,{'build_tower'})
    assert entry['server_release']=='verified-candidate' and exp.read_bytes()==original
    binding.write_text('{"different_binding":true}')
    with pytest.raises(ValueError,match='Unbound deployment correction'):
        table.record(path,summary,{'build_tower'})


def test_refresh_discovers_ignored_nested_model_runs(tmp_path, monkeypatch):
    import json
    import shutil
    import sys
    from scripts import update_robodojo_experiment_table as table
    inventory = table.ROOT / 'configs/harness/robodojo_rigid32_inventory.json'
    destination = tmp_path / 'configs/harness/robodojo_rigid32_inventory.json'
    destination.parent.mkdir(parents=True)
    shutil.copyfile(inventory, destination)
    (tmp_path / 'tmp').mkdir()
    (tmp_path / 'tmp/.ignore').write_text('runs/\n')
    run = tmp_path / 'tmp/robodojo-cross/runs/v95-seed1'
    run.mkdir(parents=True)
    (run / 'experiment.json').write_text(json.dumps({
        'env_id': 'openeta/robodojo_dual_x5_match_and_pick_from_conveyor-v0',
        'harness_version': 'v95', 'seed': 1}))
    (run / 'codex-events.jsonl').write_text('{}\n')
    (run / 'summary.json').write_text(json.dumps({
        'model': 'gpt-6-astra', 'status': 'completed',
        'host': {'tool_calls': 22, 'outcome': {
            'status': 'success', 'benchmark_success': True, 'terminal': True,
            'source': 'robodojo.eval_env.terminal_flags'}}}))
    diagnostic = tmp_path / 'tmp/robodojo-cross/preflight'
    diagnostic.mkdir()
    (diagnostic / 'summary.json').write_text('{"model":"not-a-model-run"}')
    monkeypatch.setattr(table, 'ROOT', tmp_path)
    monkeypatch.setattr(table, 'OUT', tmp_path / 'evaluations/robodojo')
    monkeypatch.setattr(sys, 'argv', ['update-table'])
    table.main()
    ledger = json.loads((table.OUT / 'experiment_ledger.json').read_text())
    assert ledger['totals']['counted_attempts'] == 1
    assert ledger['totals']['official_successes'] == 1
    assert ledger['runs'][0]['task'] == 'match_and_pick_from_conveyor'


def test_authorized_feedback_and_public_rule_do_not_merge_with_original_instruction(tmp_path,monkeypatch):
    import json
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    columns=[]
    for name,protocol,factors in [
        ('original','independent_session',{'conditioning':'zero_shot'}),
        ('rule','independent_session',{'conditioning':'zero_shot','task_rule_supplement':{'id':'mahjong_kong_replacement_v1'}}),
        ('feedback','cross_attempt_authorized_feedback',{'conditioning':'cross_attempt_authorized_feedback'})]:
        run=tmp_path/name;run.mkdir()
        (run/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_make_kong-v0','harness_version':'v103'}))
        summary={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed','evaluation_protocol':protocol,
                 'experiment_factors':factors,'host':{'tool_calls':5,'outcome':{'status':'running'}}}
        path=run/'summary.json';path.write_text(json.dumps(summary));columns.append(table.record(path,summary,{'make_kong'})['column'])
    assert len(set(columns))==3
    assert '补充公开规则' in columns[1]
    assert '跨轮判别反馈经验' in columns[2]


def test_unlimited_budget_is_separate_from_native_baseline(tmp_path,monkeypatch):
    import json
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    run=tmp_path/'model-v105';run.mkdir()
    (run/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_imitate_sorting_sequence-v0'}))
    summary={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed',
        'host':{'tool_calls':10,'simulation_budget':{'unlimited':True,'action_limit':None,
        'protocol':{'evaluation_variant':'unlimited_action_budget','effective_native_action_limit':None}},'outcome':{'status':'running'}}}
    p=run/'summary.json';p.write_text(json.dumps(summary))
    row=table.record(p,summary,{'imitate_sorting_sequence'})
    assert row['column']=='v105 · 无动作总额限制'
    assert row['native_action_limit'] is None and row['counted']==1 and row['success']==0


def test_modified_initialization_has_separate_column_and_receipt(tmp_path,monkeypatch):
    import json
    from scripts import update_robodojo_experiment_table as table
    monkeypatch.setattr(table,'ROOT',tmp_path)
    run=tmp_path/'model-v107';run.mkdir()
    (run/'experiment.json').write_text(json.dumps({'env_id':'openeta/robodojo_dual_x5_fill_egg_holder-v0'}))
    s={'model':'gpt-6-astra','reasoning_effort':'high','status':'completed',
       'host':{'tool_calls':10,'outcome':{'status':'running'}}}
    path=run/'summary.json';path.write_text(json.dumps(s))
    baseline=table.record(path,s,{'fill_egg_holder'})
    receipt={'profile':'egg_precheck_settle_300_v1','completed_steps':300,
             'native_stability_passed':True,'comparable_to_native_initialization_baseline':False}
    s['host']['initialization']=receipt;path.write_text(json.dumps(s))
    variant=table.record(path,s,{'fill_egg_holder'})
    assert variant['column']!=baseline['column']
    assert '初始化预静置' in variant['column']
    assert variant['initialization']==receipt
    assert variant['success']==0
