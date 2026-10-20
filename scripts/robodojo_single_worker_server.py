"""Start one RoboDojo worker and its MCP server, preserving the supplied GPU allocation.

Scheduler-independent version of the deployment used in the local experiments.
Requires a prepared Isaac environment and an explicit experience file.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-python", type=Path, default=ROOT / "sim/venvs/robodojo/bin/python")
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9889)
    args = parser.parse_args(argv)
    if not args.worker_python.is_file() or not args.experience.is_file():
        parser.error("Worker Python and Isaac experience must exist")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    runtime.mkdir(mode=0o700)
    os.environ.update({
        "ROBODOJO_ROBOT_CONFIG": str(ROOT / "sim/envs/robodojo/config/single_x5_left_mount.yml"),
        "OPENETA_ROBODOJO_DUAL": "1",
        "OPENETA_WORKER_POOL_MAX": "1",
        "OPENETA_ROBODOJO_SUPPORT_POLICY": "native_auxiliary_ownership_v1",
        "OPENETA_ROBODOJO_REGISTER_UNTESTED": "1",
        "OPENETA_ROBODOJO_INITIALIZATION_PROFILE": "egg_precheck_settle_300_v1",
        "OPENETA_SIM_SESSION_TTL_S": "3600",
        "OPENETA_WORKER_LOG_DIR": str(output / "worker-logs"),
        "OPENETA_EPISODE_RECORDING_DIR": str(output / "recordings"),
        "XDG_RUNTIME_DIR": str(runtime),
    })
    for key in ("ROBODOJO_DUAL_ROBOT_CONFIG", "ROBODOJO_POLICY_PORT", "ROBODOJO_POLICY_URL",
                "OPENETA_ROBODOJO_RUNTIME_PROFILE", "OPENETA_ROBODOJO_CLOTH_MATERIAL_PROFILE"):
        os.environ.pop(key, None)
    command = [str(args.worker_python.absolute()), "-u", str(ROOT / "sim/bench_worker.py"),
               "--bench", "robodojo", "--port", "0", "--headless", "--enable_cameras",
               "--experience", str(args.experience.resolve()),
               "--kit_args=--/app/extensions/registryEnabled=false"]
    log = (output / "worker.log").open("w")
    worker = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)

    def stop(*_):
        raise SystemExit(0)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        deadline = time.monotonic() + 360
        port = None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while time.monotonic() < deadline:
            if worker.poll() is not None:
                raise RuntimeError("Worker exited before becoming healthy; see worker.log")
            if port is None:
                for line in (output / "worker.log").read_text(errors="replace").splitlines():
                    if line.strip().isdigit():
                        port = int(line.strip())
                        break
            if port:
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
            time.sleep(2)
        else:
            raise TimeoutError("Worker startup exceeded 360 seconds")
        from sim.mcp_server import server
        from sim.mcp_server.worker_mgr import BenchWorkerHandle
        manager = server._get_mgr()
        manager._pools["robodojo"] = [
            BenchWorkerHandle("robodojo", port, worker, f"http://127.0.0.1:{port}")
        ]

        def no_extra_workers(bench):
            raise RuntimeError("Only the prestarted RoboDojo worker is permitted")

        manager._spawn_worker = no_extra_workers
        (output / "server-ready.json").write_text(json.dumps({
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "worker_port": port, "sim_url": f"http://{args.host}:{args.port}/mcp",
            "worker_command": command, "deployment_only": True,
        }, indent=2) + "\n")
        old_argv = sys.argv
        try:
            sys.argv = ["server", "--host", args.host, "--port", str(args.port)]
            server.main()
        finally:
            sys.argv = old_argv
    finally:
        signal.signal(signal.SIGTERM, previous)
        try:
            os.killpg(worker.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
        log.close()


if __name__ == "__main__":
    main()
