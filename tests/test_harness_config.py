import json
from pathlib import Path

import pytest

from tools.harness_config import CATALOG, ConfigError, ConfigRegistry
from tools.harness_setup import Preparation, read_selection
from scripts.codex_configured_run import parser, configure

ENV = 'openeta/libero_libero_goal_task3-v0'
TASK = {'env_id': ENV, 'task': 'Open the cabinet', 'seed': 0}


def catalog_copy(tmp_path, mutate):
    components = json.loads((CATALOG / 'components.json').read_text())
    presets = json.loads((CATALOG / 'presets.json').read_text())
    mutate(components, presets)
    (tmp_path / 'components.json').write_text(json.dumps(components))
    (tmp_path / 'presets.json').write_text(json.dumps(presets))
    return ConfigRegistry(tmp_path)


def test_frozen_config_fingerprint_and_legacy_sources():
    config = ConfigRegistry().resolve('libero_panda_v1')
    config.require_executable()
    changed = config.value
    changed['components']['execution']['controller_id'] = 'wrong'
    assert config.value['components']['execution']['controller_id'] == 'mink.robosuite_joint_velocity'
    assert config.sha256 == ConfigRegistry().resolve(config.id).sha256
    assert len(ConfigRegistry().catalog(env_id=ENV)) == 1


@pytest.mark.parametrize('id', ['../../config', '/tmp/config.json', 'libero_panda', 'missing_v1'])
def test_registered_ids_only(id):
    with pytest.raises(ConfigError):
        ConfigRegistry().resolve(id)


@pytest.mark.parametrize('change', [
    lambda c,p: c['components']['execution']['libero_mink_v1'].update(backend='robodojo'),
    lambda c,p: c['components']['execution']['libero_mink_v1'].update(scope=['base']),
    lambda c,p: c['components']['interface']['libero_atomic_v1'].update(effectors=['dexterous_hand']),
    lambda c,p: c['components']['embodiment']['panda_parallel_v1']['effectors']['primary'].update(group={}),
    lambda c,p: c['components']['embodiment']['panda_parallel_v1']['effectors']['primary'].update(tcp_rotation_from_reported=[[1,0,0],[0,1,0],[0,0,-1]]),
    lambda c,p: c['components']['embodiment']['panda_parallel_v1']['effectors']['primary'].update(tcp_rotation_from_reported=['oops']),
    lambda c,p: c['components']['embodiment']['panda_parallel_v1']['effectors']['primary'].update(calibration=None),
    lambda c,p: c['components']['execution']['libero_mink_v1'].update(implemented=1),
    lambda c,p: c['components']['execution']['libero_mink_v1'].update(disable_collision_check=True),
    lambda c,p: p['presets'].append(p['presets'][0]),
    lambda c,p: c.update(schema_version=True),
])
def test_invalid_compositions_fail_closed(tmp_path, change):
    with pytest.raises(ConfigError):
        catalog_copy(tmp_path, change)


def test_duplicate_json_keys_rejected(tmp_path):
    (tmp_path / 'components.json').write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ConfigError, match='Duplicate key'):
        ConfigRegistry(tmp_path)


def test_legacy_factory_cannot_silently_ignore_a_new_calibration(tmp_path):
    registry = catalog_copy(tmp_path, lambda c,p:
        c['components']['embodiment']['panda_parallel_v1']['effectors']['primary'].update(
            tcp_rotation_from_reported=[[1,0,0],[0,1,0],[0,0,1]]))
    with pytest.raises(ConfigError) as error:
        registry.resolve('libero_panda_v1').require_executable()
    assert error.value.code == 'unsupported_legacy_configuration'


def test_unimplemented_robot_not_selectable(tmp_path):
    registry = ConfigRegistry()
    config = registry.resolve('robodojo_dual_x5_v1')
    assert 'x5_world_self_interarm_collision' in config.public()['missing_capabilities']
    prep = Preparation(env_id='openeta/robodojo_dual_x5_stack_blocks-v0', task='Stack blocks', seed=0, output=tmp_path/'selection.json')
    with pytest.raises(ConfigError) as error:
        prep.select(config.id, config.sha256)
    assert error.value.code == 'config_not_executable'
    assert not prep.output.exists()


def test_binding_selection_idempotent_and_tied_to_exact_task(tmp_path):
    prep = Preparation(**TASK, output=tmp_path/'selection.json')
    config = ConfigRegistry().resolve('libero_panda_v1')
    first = prep.select(config.id, config.sha256)
    first['binding']['task_instance']['seed'] = 99
    assert prep.select(config.id, config.sha256)['binding']['task_instance']['seed'] == 0
    assert read_selection(prep.output, **TASK).sha256 == config.sha256
    with pytest.raises(ConfigError) as error:
        read_selection(prep.output, **{**TASK, 'seed': 1})
    assert error.value.code == 'selection_binding_mismatch'
    other = Preparation(**TASK, output=prep.output)
    with pytest.raises(ConfigError) as error:
        other.select(config.id, config.sha256)
    assert error.value.code == 'selection_output_exists'


def test_selection_rejects_stale_hash_and_tampered_artifact(tmp_path):
    prep = Preparation(**TASK, output=tmp_path/'selection.json')
    with pytest.raises(ConfigError) as error:
        prep.select('libero_panda_v1', '0'*64)
    assert error.value.code == 'config_hash_mismatch'
    assert not prep.output.exists()
    config = ConfigRegistry().resolve('libero_panda_v1')
    prep.select(config.id, config.sha256)
    data = json.loads(prep.output.read_text())
    data['binding_sha256'] = 'f'*64
    prep.output.write_text(json.dumps(data))
    with pytest.raises(ConfigError):
        read_selection(prep.output, **TASK)


def test_new_launcher_defaults_and_legacy_defaults_are_separate(tmp_path):
    from scripts.codex_plugin_smoke import parser as old_parser
    common = ['--sim-url', 'http://localhost:1/sse', '--task', TASK['task'], '--env-id', ENV, '--output', str(tmp_path/'run')]
    args = configure(parser().parse_args([*common, '--config-id', 'libero_panda_v1']))
    assert args.tool_profile == 'atomic' and args.controller == 'mink_joint_velocity'
    assert args.env_id == ENV and args.seed == 0
    assert old_parser().parse_args(common).tool_profile == 'pick'
    with pytest.raises(ConfigError) as error:
        configure(parser().parse_args([*common, '--config-id', 'libero_panda_v1', '--controller', 'osc_pose']))
    assert error.value.code == 'config_argument_conflict'


def test_launcher_consumes_preparation_selection(tmp_path):
    prep = Preparation(**TASK, output=tmp_path/'selection.json')
    config = ConfigRegistry().resolve('libero_panda_v1')
    prep.select(config.id, config.sha256)
    args = parser().parse_args(['--sim-url', 'http://localhost:1/sse', '--task', TASK['task'], '--env-id', ENV,
                              '--output', str(tmp_path/'run'), '--selection', str(prep.output)])
    assert configure(args).harness_config == config.id


def test_gpu_runtime_is_separate_from_cpu_profile():
    cpu=ConfigRegistry().resolve('robodojo_dual_x5_capabilities_v72')
    gpu=ConfigRegistry().resolve('robodojo_dual_x5_gpu_v73')
    assert 'simulation_profile' not in cpu.value['components']['execution']
    assert gpu.value['components']['execution']['simulation_profile']=='gpu_tensor_v1'
    assert cpu.sha256!=gpu.sha256
    cpu.require_executable()
    with pytest.raises(ConfigError,match='does not support'):
        gpu.bind_task(env_id='openeta/robodojo_dual_x5_store_laptop_and_headphones-v0',task='diagnostic',seed=0)
    gpu.require_executable()


def test_unreviewed_gpu_profile_rejected(tmp_path):
    with pytest.raises(ConfigError,match='simulation profile'):
        catalog_copy(tmp_path,lambda c,p:c['components']['execution']['robodojo_gpu_contact_dual_v73'].update(simulation_profile='anything'))


@pytest.mark.parametrize('task',['make_kong','imitate_sorting_sequence','play_tic_tac_toe'])
def test_auxiliary_configs_keep_only_two_agent_effectors(task):
    config=ConfigRegistry().resolve('robodojo_dual_x5_'+task+'_auxiliary_v99')
    config.require_executable()
    body=config.value['components']['embodiment']
    assert set(body['effectors'])=={'left','right'}
    assert body['support_actors']['support_arm0']['agent_controllable'] is False
    assert config.bind_task(env_id='openeta/robodojo_dual_x5_'+task+'-v0',task='Public instruction',seed=0)


@pytest.mark.parametrize('change',[
    lambda c,p:c['components']['execution']['robodojo_native_auxiliary_v99'].pop('support_policy'),
    lambda c,p:c['components']['embodiment']['dual_x5_native_franka_v99']['support_actors']['support_arm0'].update(agent_controllable=True),
    lambda c,p:c['components']['embodiment']['dual_x5_native_franka_v99'].update(native_layout='dual_x5'),
])
def test_auxiliary_config_requires_coherent_control_ownership(tmp_path,change):
    config=catalog_copy(tmp_path,change).resolve('robodojo_dual_x5_make_kong_auxiliary_v99')
    with pytest.raises(ConfigError,match='ownership'):config.require_executable()


@pytest.mark.parametrize('task,override',[
    ('make_kong',None),
    ('play_tic_tac_toe','play_tic_tac_toe_native_adapter_v101'),
    ('imitate_sorting_sequence','imitate_sorting_sequence_native_adapter_v101'),
])
@pytest.mark.parametrize('version',['v101','v102'])
def test_progress_guard_presets_bind_only_explicit_native_adapter_alignment(task,override,version):
    config=ConfigRegistry().resolve('robodojo_dual_x5_'+task+'_auxiliary_'+version)
    config.require_executable()
    assert config.value['components']['benchmark'].get('episode_budget_override')==override
    assert set(config.value['components']['embodiment']['effectors'])=={'left','right'}
    assert ConfigRegistry().resolve('robodojo_dual_x5_'+task+'_auxiliary_v100').value['components']['benchmark'].get('episode_budget_override') is None


@pytest.mark.parametrize('task',['make_kong','play_tic_tac_toe','imitate_sorting_sequence'])
def test_v105_unlimited_presets_preserve_task_and_ownership(task):
    config=ConfigRegistry().resolve('robodojo_dual_x5_'+task+'_auxiliary_v105')
    config.require_executable()
    assert config.value['components']['benchmark']['episode_budget_override']=='robodojo_unlimited_actions_v105'
    assert set(config.value['components']['embodiment']['effectors'])=={'left','right'}
    assert ConfigRegistry().resolve('robodojo_dual_x5_'+task+'_auxiliary_v103').value['components']['benchmark'].get('episode_budget_override')!='robodojo_unlimited_actions_v105'

@pytest.mark.parametrize('task',['make_kong','play_tic_tac_toe','imitate_sorting_sequence'])
def test_configured_robodojo_tool_budget_defaults_and_historical_override(task):
    base='robodojo_dual_x5_'+task+'_auxiliary_v103'
    argv=['--sim-url','http://example.invalid/mcp','--config-id',base,'--env-id','openeta/robodojo_dual_x5_'+task+'-v0',
          '--task','test','--output','/tmp/not-executed']
    args=configure(parser().parse_args(argv))
    assert args.max_requests==200 and args.timeout==0
    resolved=ConfigRegistry().resolve(args.harness_config)
    assert resolved.sha256==args.harness_config_sha256
    assert resolved.value['components']['benchmark']['episode_budget_override']=='robodojo_unlimited_actions_v105'
    args=configure(parser().parse_args(argv+['--episode-action-budget','profile','--timeout','10800','--max-requests','80']))
    assert args.harness_config==base and args.timeout==10800 and args.max_requests==80


def test_unlimited_variant_supports_single_arm_without_changing_libero():
    registry=ConfigRegistry();base=registry.resolve('robodojo_single_x5_left_mount_v1')
    new=registry.resolve(base.id+'__unlimited_v105');new.require_executable()
    assert new.value['components']['embodiment']==base.value['components']['embodiment']
    with pytest.raises(ConfigError):registry.resolve('libero_panda_v1__unlimited_v105')
