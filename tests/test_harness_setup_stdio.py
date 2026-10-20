"""Real MCP handshake and preparation artifact consumed by the run launcher."""
import asyncio
import json
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from scripts.codex_configured_run import configure, parser


def test_preparation_real_stdio(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'selection.json'
    env_id = 'openeta/libero_libero_object_task0-v0'
    task = 'Pick up the object'

    async def check():
        params = StdioServerParameters(command=sys.executable,
            args=['-m', 'tools.harness_setup', '--env-id', env_id, '--task', task, '--output', str(output)],
            env={**os.environ, 'PYTHONPATH': str(root), 'PYTHONDONTWRITEBYTECODE': '1'}, cwd=str(root))
        async with asyncio.timeout(30):
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    names = {t.name for t in (await client.list_tools()).tools}
                    assert names == {'list_harness_configs', 'select_harness_config'}
                    response = await client.call_tool('list_harness_configs', {})
                    catalog = json.loads(response.content[0].text)
                    assert len(catalog['configs']) == 1
                    config = catalog['configs'][0]
                    rejected = await client.call_tool('select_harness_config', {'config_id': config['id'], 'config_sha256': 'stale'})
                    assert json.loads(rejected.content[0].text)['code'] == 'config_hash_mismatch'
                    assert not output.exists()
                    reply = await client.call_tool('select_harness_config', {'config_id': config['id'], 'config_sha256': config['sha256']})
                    assert json.loads(reply.content[0].text)['ok'] is True
                    again = await client.call_tool('select_harness_config', {'config_id': config['id'], 'config_sha256': config['sha256']})
                    assert json.loads(again.content[0].text)['idempotent'] is True

    asyncio.run(check())
    audit = [json.loads(line) for line in output.with_suffix('.mcp.jsonl').read_text().splitlines()]
    assert [entry['tool'] for entry in audit] == ['list_harness_configs'] + ['select_harness_config']*3
    assert [entry['code'] for entry in audit] == ['ok','config_hash_mismatch','ok','ok']
    args = parser().parse_args(['--selection', str(output), '--sim-url', 'http://localhost:1/sse',
                              '--env-id', env_id, '--task', task, '--output', str(tmp_path/'run')])
    assert configure(args).harness_config == 'libero_panda_v1'
