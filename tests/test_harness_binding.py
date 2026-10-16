from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_codex_atomic import atomic
from test_codex_motion_hook import rig
from tools.harness_binding import build_configured_host
from tools.harness_config import ConfigError, ConfigRegistry

ENV = 'openeta/libero_libero_goal_task3-v0'


def args():
    return Namespace(harness_config='libero_panda_v1', harness_config_sha256=None,
                     env_id=ENV, task='Open the cabinet', seed=0, tool_profile='atomic',
                     expected_controller='mink_joint_velocity')


def advertised():
    return {'backend': 'libero', 'env_id': ENV, 'robot': '', 'control_spec': {'controller': {
        'controller_id': 'mink.robosuite_joint_velocity', 'supports_position': True, 'supports_orientation': True}}}


def fake_host(tmp_path):
    closed = []
    return SimpleNamespace(output=tmp_path, runner=SimpleNamespace(environment=SimpleNamespace(create_result=advertised())),
        runtime=SimpleNamespace(memory=SimpleNamespace(controller_capabilities=lambda: {
            'controller_id': 'mink.robosuite_joint_velocity', 'motion_owns_trajectory_world_collision': True})),
        close=lambda reason: closed.append(reason), closed=closed)


@pytest.mark.parametrize('field,value', [('harness_config_sha256', 'wrong'), ('env_id', 'robodojo'),
                                        ('tool_profile', 'pick'), ('expected_controller', 'osc_pose')])
def test_preflight_failure_never_creates_environment(field, value):
    value_args = args()
    setattr(value_args, field, value)
    calls = []
    with pytest.raises(ConfigError):
        build_configured_host(value_args, legacy_builder=lambda a: calls.append(a))
    assert calls == []


@pytest.mark.parametrize('change', [
    lambda h: h.runner.environment.create_result.update(backend='robodojo'),
    lambda h: h.runner.environment.create_result.update(env_id='wrong'),
    lambda h: h.runner.environment.create_result.update(robot='X5'),
    lambda h: h.runner.environment.create_result['control_spec']['controller'].update(controller_id='wrong'),
    lambda h: h.runner.environment.create_result['control_spec']['controller'].update(supports_orientation=1),
    lambda h: setattr(h.runtime.memory, 'controller_capabilities', lambda: {'controller_id': 'mink.robosuite_joint_velocity'}),
])
def test_live_mismatch_closes_environment_before_exposure(tmp_path, change):
    host = fake_host(tmp_path)
    change(host)
    with pytest.raises(ConfigError):
        build_configured_host(args(), legacy_builder=lambda a: host)
    assert host.closed == ['harness_binding_failed']
    assert not (tmp_path/'harness-binding.json').exists()


def test_legacy_build_dispatch_untouched(monkeypatch):
    import tools.codex_host as module
    original = Namespace()
    calls = []
    monkeypatch.setattr(module, '_build_legacy_host', lambda a: calls.append(a) or 'legacy')
    assert module.build_host(original) == 'legacy'
    assert calls == [original]


def test_legacy_builder_body_matches_checkpoint():
    import ast
    import hashlib
    import inspect
    from tools.codex_host import _build_legacy_host
    node = ast.parse(inspect.getsource(_build_legacy_host)).body[0]
    fingerprint = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
    golden = json.loads((Path(__file__).parent/'fixtures/harness/libero_atomic_schema_9d9b5a8.json').read_text())
    assert fingerprint == golden['legacy_builder_ast_sha256']


def test_binding_preserves_six_tool_schemas_and_motion_hook(atomic, monkeypatch):
    host, state = atomic
    before = {k: v.model_dump() for k,v in host.schemas.items()}
    golden = json.loads((Path(__file__).parent/'fixtures/harness/libero_atomic_schema_9d9b5a8.json').read_text())
    assert before == golden['schemas']
    host.runner.environment.create_result = advertised()
    monkeypatch.setattr(host.runtime.memory, 'controller_capabilities', lambda: {
        'controller_id': 'mink.robosuite_joint_velocity', 'motion_owns_trajectory_world_collision': True})
    result = build_configured_host(args(), legacy_builder=lambda a: host)
    assert result is host
    assert {k: v.model_dump() for k,v in result.schemas.items()} == before
    record = json.loads((host.output/'harness-binding.json').read_text())
    assert record['config_sha256'] == ConfigRegistry().resolve('libero_panda_v1').sha256
    assert not record['model_interface_changed']
    reply = result.call('move_to', {'delta_m': [0, 0, .02]})
    assert not reply.isError, reply
    assert [n for n,p in state['calls']] == ['ik_preview_check', 'move_to']
    assert state['calls'][-1][1]['enable_collision_check'] is True
    assert result.requests == 1 and result.runner.tool_call_count == 3
    assert 'config_sha256' not in reply.content[0].text
