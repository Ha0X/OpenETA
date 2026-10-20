"""Check client/server composition of the public export without simulator actions."""
import asyncio
import json
from pathlib import Path


def test_all_v112_presets_keep_executable_bindings_and_register_dual_ids(monkeypatch):
    monkeypatch.setenv('OPENETA_ROBODOJO_DUAL', '1')
    monkeypatch.setenv('OPENETA_ROBODOJO_REGISTER_UNTESTED', '1')
    import sim.env_registry as envs
    from tools.harness_config import ConfigRegistry
    from sim.mcp_server.worker_mgr import _bench_for_env_id
    envs._register_robodojo_dual_envs()
    rows = json.loads(Path('configs/harness/presets.json').read_text())['presets']
    ids = [row['id'] for row in rows if row['id'].endswith('_repair_v112')]
    assert len(ids) == 15
    for name in ids:
        config = ConfigRegistry().resolve(name)
        config.require_executable()
        eid = config.value['components']['benchmark']['env_pattern']
        assert _bench_for_env_id(eid) == 'robodojo'
        assert envs.get_env_spec(eid).env_type == 'robodojo'


def test_server_registers_configured_host_transport():
    from sim.mcp_server import server
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {'configured_robodojo', 'create_env', 'reset_env', 'close_env'} <= names


def test_published_curve_is_recomputed_from_public_attempt_ledger():
    ledger = json.loads(Path('docs/assets/robodojo-rsi-progress-20261016.json').read_text())
    assert len(ledger['tasks']) == ledger['denominator'] == 42
    for point in ledger['points']:
        n = point['attempt_budget']
        assert point['successes'] == sum(any(r['official_success'] for r in rows[:n])
                                        for rows in ledger['tasks'].values())
    assert ledger['provisional'] is True
