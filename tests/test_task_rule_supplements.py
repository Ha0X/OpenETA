import hashlib
import pytest
from tools.task_rule_supplements import task_rule_supplement, KONG_ENV, KONG_RULE, KONG_RULE_ID
from tools.harness_config import ConfigRegistry, ConfigError, _validate_component


def test_new_preset_delivers_public_rule_and_records_changed_instruction_condition():
    registry=ConfigRegistry();c=registry.resolve('robodojo_dual_x5_make_kong_auxiliary_v103');c.require_executable()
    b=c.value['components']['benchmark'];prompt,record=task_rule_supplement(b,KONG_ENV)
    assert KONG_RULE in prompt and 'replacement tile' in prompt
    assert record['text_sha256']==hashlib.sha256(KONG_RULE.encode()).hexdigest()
    assert record['comparable_to_original_instruction'] is False
    assert record['source_section']=='3.6.8'
    assert 'mahjong9' not in prompt and '0.319' not in prompt and 'tolerance' not in prompt
    old=registry.resolve('robodojo_dual_x5_make_kong_auxiliary_v102').value['components']['benchmark']
    assert task_rule_supplement(old,KONG_ENV)==('',None)
    assert task_rule_supplement({},'openeta/libero_libero_goal_task0-v0')==('',None)


@pytest.mark.parametrize('mutation', [dict(task_rule_supplement='unknown'),dict(task_rule_supplement=None),dict(backend='libero'),dict(env_pattern='openeta/robodojo_dual_x5_play_tic_tac_toe-v0')])
def test_supplement_cannot_silently_apply_to_other_tasks(mutation):
    c=ConfigRegistry().resolve('robodojo_dual_x5_make_kong_auxiliary_v103')
    b=dict(c.value['components']['benchmark']);b.update(mutation)
    with pytest.raises(ConfigError) as exc:
        _validate_component('benchmark',b['id'],b)
    assert exc.value.code == 'invalid_task_rule_supplement'
    if mutation.get('task_rule_supplement','notnull') is not None:
        with pytest.raises(ValueError):task_rule_supplement(b,KONG_ENV)


@pytest.mark.parametrize('task', ['press_by_number', 'organize_table'])
def test_instruction_corrections_are_opt_in_and_bound_to_exact_task(task):
    registry = ConfigRegistry()
    c = registry.resolve('robodojo_dual_x5_' + task + '_repair_v109')
    c.require_executable()
    b = c.value['components']['benchmark']
    prompt, record = task_rule_supplement(b, b['env_pattern'])
    assert record['evaluation_condition'] == 'task_instruction_corrected'
    assert record['comparable_to_original_instruction'] is False
    assert record['text'] in prompt
    assert record['text_sha256'] == hashlib.sha256(record['text'].encode()).hexdigest()
    with pytest.raises(ValueError):
        task_rule_supplement(b, KONG_ENV)
    legacy = dict(b)
    legacy.pop('task_rule_supplement')
    assert task_rule_supplement(legacy, b['env_pattern']) == ('', None)



def test_table_v110_clarifies_all_requirements_without_rewriting_v109():
    from tools.task_rule_supplements import CORRECTED_RULES
    c=ConfigRegistry().resolve('robodojo_dual_x5_organize_table_repair_v110');c.require_executable()
    b=c.value['components']['benchmark'];prompt,record=task_rule_supplement(b,b['env_pattern'])
    for phrase in ('mouse pad','world +X','world +Y', 'upright','centered','resting stably','initial positions and orientations','pending','static fixture'):
        assert phrase in prompt
    for private in ('0.225','0.002','15 degrees','45 degrees','rect_bounds','cube_cushion','garage'):
        assert private not in prompt
    old=CORRECTED_RULES['organize_table_instruction_correction_v109']['text']
    assert 'local +Y' not in old and 'upright' not in old
    assert record['source'].endswith('requirements-v110-20261012.md')
    assert record['evaluation_condition']=='task_instruction_corrected'
    assert not record['comparable_to_original_instruction']
    with pytest.raises(ValueError):task_rule_supplement(b,KONG_ENV)


def test_v111_all_18_retry_presets_preserve_original_task_binding_and_budgets():
    from tools.task_rule_supplements import TASK_REQUIREMENTS_V111
    registry=ConfigRegistry()
    assert len(TASK_REQUIREMENTS_V111)==17
    for task in [*TASK_REQUIREMENTS_V111,'organize_table']:
        c=registry.resolve('robodojo_dual_x5_'+task+'_repair_v111');c.require_executable()
        old=registry.resolve('robodojo_dual_x5_'+task+'_batch_v108')
        b=c.value['components']['benchmark'];prompt,record=task_rule_supplement(b,b['env_pattern'])
        assert b['env_pattern']==old.value['components']['benchmark']['env_pattern']
        assert b['episode_budget_override']=='robodojo_unlimited_actions_v105'
        assert record['evaluation_condition']=='task_instruction_corrected'
        assert record['text_sha256']==hashlib.sha256(record['text'].encode()).hexdigest()
        assert prompt and not record['comparable_to_original_instruction']
        for forbidden in ('dis_threshold','qpos','mahjong9','0.319','0.002','0.95','15 degrees'):
            assert forbidden not in prompt
        assert c.value['components']['execution']==old.value['components']['execution']
