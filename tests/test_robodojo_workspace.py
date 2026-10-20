"""Single-checkout launch checks; no model, simulator process or GPU required."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def test_openeta_defaults_to_packaged_host_and_reads_its_guidance():
    from scripts.robodojo_openeta_run import parser, load_reference_guidance
    preset = "robodojo_dual_x5_pour_balls_into_vase_repair_v112"
    env_id = "openeta/robodojo_dual_x5_pour_balls_into_vase-v0"
    args = parser().parse_args([
        "--sim-url", "http://127.0.0.1:9889/mcp", "--config-id", preset,
        "--env-id", env_id, "--task-file", "task.txt", "--output", "outputs/new",
    ])
    assert args.robodojo_root == Path(__file__).resolve().parents[1]
    guidance = load_reference_guidance(root=args.robodojo_root, python=sys.executable,
                                      config_id=preset, env_id=env_id)
    assert guidance["config_sha256"]
    assert guidance["workflow"]


def test_single_worker_inherits_allocation_and_cleans_up(tmp_path, monkeypatch):
    from scripts import robodojo_single_worker_server as launcher
    from sim.mcp_server import server
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-assigned-by-allocator")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    experience = tmp_path / "test.kit"
    experience.write_text("")
    output = tmp_path / "server"
    stopped = []
    worker = SimpleNamespace(pid=12345, poll=lambda: None,
                             wait=lambda **kw: stopped.append("wait"))

    def start(command, **kwargs):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-assigned-by-allocator"
        assert command[0] == str(Path(sys.executable).absolute())
        assert command[command.index("--experience") + 1] == str(experience)
        kwargs["stdout"].write("29999\n")
        kwargs["stdout"].flush()
        return worker

    class Health:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    manager = SimpleNamespace(_pools={})
    monkeypatch.setattr(launcher.subprocess, "Popen", start)
    monkeypatch.setattr(launcher.os, "killpg", lambda *args: stopped.append("signal"))
    monkeypatch.setattr(launcher.urllib.request, "build_opener",
                        lambda *args: SimpleNamespace(open=lambda *a, **kw: Health()))
    monkeypatch.setattr(server, "_get_mgr", lambda: manager)

    def serve():
        assert manager._pools["robodojo"][0].port == 29999
        with pytest.raises(RuntimeError, match="prestarted"):
            manager._spawn_worker("robodojo")
        assert sys.argv == ["server", "--host", "127.0.0.1", "--port", "9889"]

    monkeypatch.setattr(server, "main", serve)
    launcher.main(["--worker-python", sys.executable, "--experience", str(experience),
                   "--output", str(output)])
    ready = json.loads((output / "server-ready.json").read_text())
    assert ready["cuda_visible_devices"] == "GPU-assigned-by-allocator"
    assert stopped == ["signal", "wait"]
