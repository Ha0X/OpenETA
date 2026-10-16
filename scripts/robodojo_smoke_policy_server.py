"""Minimal real XPolicyLab websocket policy endpoint for RoboDojo smoke."""
import argparse
import asyncio
import os
import sys
import types
from pathlib import Path

# Isaac can exit without unwinding the worker's Python cleanup. Tie this
# child to its parent in the kernel so close_env cannot orphan the policy.
if sys.platform == "linux":
    import ctypes
    import signal

    ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
    if os.getppid() == 1:
        raise SystemExit(0)

sys.path.insert(0, str(Path(os.environ["ROBODOJO_ROOT"]) / "XPolicyLab"))
try:
    import websockets.asyncio.server  # type: ignore
except ModuleNotFoundError:
    import websockets.server as _legacy_server
    _legacy_server.Server = _legacy_server.WebSocketServer
    _asyncio_pkg = types.ModuleType("websockets.asyncio")
    _asyncio_pkg.server = _legacy_server
    sys.modules["websockets.asyncio"] = _asyncio_pkg
    sys.modules["websockets.asyncio.server"] = _legacy_server
from client_server.ws.model_server import PolicyServer, PolicyServerConfig


class SmokeModel:
    def reset(self): return None
    def update_obs(self, obs): return None
    def update_obs_batch(self, obs_list): return None
    def get_action(self): return []
    def get_action_batch(self, env_idx_list=None): return [[] for _ in (env_idx_list or [0])]
    def prepare_case(self, case_meta=None): return None
    def on_trial_end(self, result=None): return None


async def main(port: int) -> None:
    server = PolicyServer(SmokeModel(), PolicyServerConfig(host="127.0.0.1", port=port))
    await server.start()
    print(f"[smoke-policy] listening on ws://127.0.0.1:{port}", flush=True)
    try:
        await asyncio.Future()
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    asyncio.run(main(parser.parse_args().port))
