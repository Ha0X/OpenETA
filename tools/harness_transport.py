"""Streamable HTTP transport for configured simulator integrations.

The legacy LIBERO SSE transport is unchanged. Each operation owns its MCP
connection; callers must carry the simulator's application session_id and
handle across calls. A transport failure is never retried automatically:
an action might already have executed on the server.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent.tools.sim_mcp import (
    SimulatorMcpTransportError,
    _mcp_host_http_client_factory,
    _parse_mcp_tool_result,
    _parse_mcp_tools_result,
    _timeout_delta,
    _with_optional_timeout,
)


@dataclass(frozen=True, slots=True)
class StreamableHttpSimulatorMcpTransport:
    url: str

    def list_tools(self, *, timeout_s: float | None = None) -> dict:
        return self._run(None, {}, timeout_s)

    def call_tool(self, name: str, arguments: dict, *, timeout_s: float | None = None) -> dict:
        return self._run(name, arguments, timeout_s)

    def _run(self, name, arguments, timeout_s):
        operation = 'list_tools' if name is None else f'call_tool:{name}'
        try:
            return asyncio.run(_with_optional_timeout(
                self._request(name, arguments, timeout_s), timeout_s=timeout_s))
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError(operation, exc) from exc

    async def _request(self, name, arguments, timeout_s):
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        # Bypass proxies for this MCP host only; retain the existing CA and
        # proxy configuration for other hosts, including redirect targets.
        timeout = timeout_s if timeout_s is not None and timeout_s > 0 else 300.0
        factory = _mcp_host_http_client_factory(self.url)
        async with factory(timeout=httpx.Timeout(timeout, connect=min(timeout, 30.0))) as client:
            async with streamable_http_client(self.url, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if name is not None:
                        result = await session.call_tool(name, arguments,
                            read_timeout_seconds=_timeout_delta(timeout_s))
                        return _parse_mcp_tool_result(result)
                    rows, cursor, seen = [], None, set()
                    while True:
                        result = await session.list_tools(cursor=cursor)
                        rows.extend(_parse_mcp_tools_result(result)['tools'])
                        cursor = result.nextCursor
                        if not cursor:
                            return {'tools': rows, 'tool_count': len(rows)}
                        if cursor in seen:
                            raise ValueError('MCP tools/list repeated its pagination cursor')
                        seen.add(cursor)
