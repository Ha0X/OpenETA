"""Exercise actual Streamable HTTP framing, errors and no-retry semantics."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from agent.tools.sim_mcp import SimulatorMcpTransportError
from tools.harness_transport import StreamableHttpSimulatorMcpTransport


@pytest.fixture
def remote(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    server = tmp_path/'server.py'
    server.write_text('''
import asyncio, sys
from mcp.server.fastmcp import FastMCP
mcp = FastMCP('transport-test', host='127.0.0.1', port=int(sys.argv[1]))
counts = {}
@mcp.tool()
async def command(session_id: str, handle: str, delay: float = 0.0) -> dict:
    counts[session_id] = counts.get(session_id, 0) + 1
    await asyncio.sleep(delay)
    return {'session_id': session_id, 'handle': handle, 'count': counts[session_id]}
@mcp.tool()
def inspect_session(session_id: str) -> dict:
    return {'count': counts.get(session_id, 0)}
@mcp.tool()
def rejected() -> dict:
    raise ValueError('movement refused by test server')
mcp.run(transport='streamable-http')
''')
    root = Path(__file__).resolve().parents[1]
    log = (tmp_path/'server.log').open('w')
    process = subprocess.Popen([sys.executable, str(server), str(port)], cwd=root,
        env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'}, stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+15
        while True:
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    break
            except OSError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    pytest.fail((tmp_path/'server.log').read_text())
                time.sleep(.05)
        yield StreamableHttpSimulatorMcpTransport(f'http://127.0.0.1:{port}/mcp')
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


def test_streamable_catalog_and_application_session_survive_connection_close(remote, monkeypatch):
    # Endpoint-specific bypass must work even with an unusable default proxy.
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('ALL_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('NO_PROXY', '')
    tools = remote.list_tools(timeout_s=5)
    assert {t['name'] for t in tools['tools']} == {'command','inspect_session','rejected'}
    assert tools['tool_count'] == 3
    first = remote.call_tool('command', {'session_id':'owned', 'handle':'arm'}, timeout_s=5)
    second = remote.call_tool('command', {'session_id':'owned', 'handle':'arm'}, timeout_s=5)
    assert first['count'] == 1 and second['count'] == 2
    assert second['session_id'] == 'owned' and second['handle'] == 'arm'
    assert remote.call_tool('inspect_session', {'session_id':'other'}, timeout_s=5)['count'] == 0


def test_streamable_timeout_does_not_repeat_dispatched_action(remote):
    with pytest.raises(SimulatorMcpTransportError):
        remote.call_tool('command', {'session_id':'timeout', 'handle':'arm', 'delay':2.0}, timeout_s=.8)
    assert remote.call_tool('inspect_session', {'session_id':'timeout'}, timeout_s=5)['count'] == 1


def test_streamable_tool_error_keeps_failure(remote):
    reply = remote.call_tool('rejected', {}, timeout_s=5)
    assert reply['success'] is False
    assert reply['details']['mcp_is_error'] is True
    assert 'movement refused' in str(reply)
