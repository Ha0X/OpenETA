"""Run one RoboDojo task using main's OpenETA planner and episode loop.

Example (requires an already-running RoboDojo simulator MCP service)::

    python -m scripts.robodojo_openeta_run --robodojo-root ../OpenETA-robodojo \
      --sim-url http://COMPUTE_NODE:PORT/mcp --config-id PRESET \
      --env-id ENV_ID --task-file TASK_FILE --output outputs/robodojo-openeta-one

Model credentials use the existing OpenETA provider configuration. The sibling
worktree supplies only its checked task Host; no Codex login or CLI is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from agent.backends.planner import OpenAICompatiblePlannerBackend
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.skills import SkillRegistry
from agent.tools.contracts import build_default_tool_contract_catalog
from agent.tools.robodojo import RoboDojoEpisodeEnvironment, RoboDojoHostConnection, RoboDojoTools


ENVIRONMENT_INSTRUCTIONS = """
This episode uses RoboDojo's checked task Host. Begin with observe or
robodojo_episode_status to obtain the current images and robot state. Use only
the advertised tools. Packaged guidance uses native names: map move_to to
robodojo_move_to, gripper_control to robodojo_gripper_control, and likewise for
the other robodojo_ tools; observe keeps its name. Both arms are exposed by role.
The movement tool owns IK and collision checks internally. Its continuous
openness command is distinct from the measured opening. Read optional packaged
guidance with robodojo_read_skill. Labelled derived images are overlays or crops
of public camera frames, not new camera measurements; mark_point uses original
image pixels. robodojo_check_task_completion provides feedback; only the official
outcome can establish task success. Do not infer success from a completed call.
"""


def build_runtime(bridge, backend, *, memory_root: Path, context_window_tokens=None):
    catalog = build_default_tool_contract_catalog(bridge.registry.list())
    return OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            backend, tool_contract_catalog=catalog,
            environment_instructions=ENVIRONMENT_INSTRUCTIONS,
            context_config=PlannerContextConfig(
                max_current_images=8, context_window_tokens=context_window_tokens,
            ),
        ),
        pipeline=ActionPipeline(tool_contract_catalog=catalog),
        tools=bridge.registry, skills=SkillRegistry(),
        memory_store=JsonMemoryStore(memory_root),
    )


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robodojo-root", type=Path, required=True)
    p.add_argument("--host-python", type=Path, help="Defaults to the reference worktree .venv/bin/python")
    p.add_argument("--sim-url", required=True)
    p.add_argument("--config-id", required=True)
    p.add_argument("--config-sha256")
    p.add_argument("--env-id", required=True)
    p.add_argument("--task-file", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", help="Override the existing OpenETA provider's model")
    p.add_argument("--max-turns", type=int, default=80)
    p.add_argument("--max-tool-calls", type=int, default=80)
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--max-total-tokens", type=int, default=5_000_000)
    p.add_argument("--output", type=Path, required=True)
    return p


def host_command(args, *, root: Path, output: Path, task: str):
    command = str((args.host_python or root / ".venv/bin/python").resolve())
    argv = [
        "-m", "tools.codex_mcp_server", "--harness-config", args.config_id,
        "--sim-url", args.sim_url, "--env-id", args.env_id, "--task", task,
        "--seed", str(args.seed), "--tool-profile", "atomic",
        "--expected-controller", "robodojo_checked", "--output", str(output / "host"),
        "--timeout", str(args.timeout), "--max-requests", str(args.max_tool_calls),
    ]
    if args.config_sha256:
        argv += ["--harness-config-sha256", args.config_sha256]
    return command, argv


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.max_turns, args.max_tool_calls, args.timeout, args.max_total_tokens) <= 0:
        p.error("Episode budgets must be positive")
    root, output = args.robodojo_root.resolve(), args.output.resolve()
    if not (root / "tools/codex_mcp_server.py").is_file():
        p.error("--robodojo-root must contain the checked RoboDojo Host")
    task = args.task_file.read_text(encoding="utf-8").strip()
    if not task:
        p.error("Task file must contain the official task instruction")
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    if provider.missing_fields():
        p.error("Missing OpenETA provider fields: " + ", ".join(provider.missing_fields()))
    command, host_args = host_command(args, root=root, output=output, task=task)
    if not Path(command).is_file():
        p.error("Host Python does not exist")
    output.mkdir(parents=True, exist_ok=False)
    host_revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
    ).strip()
    run_metadata = {
        "source": "RoboDojoOpenEtaRun", "require_official_reward": True,
        "model": provider.model, "provider": provider.provider,
        "config_id": args.config_id, "env_id": args.env_id, "seed": args.seed,
        "reference_worktree": str(root), "reference_revision": host_revision,
        "main_revision": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "harness": "OpenEtaAgentRuntime + ToolCallingPlanner + OpenEtaEpisodeRunner",
    }
    (output / "run.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    connection = bridge = None
    summary = {**run_metadata, "status": "failed", "task_success": False}
    try:
        connection = RoboDojoHostConnection(
            command=command, args=host_args, cwd=root, stderr_path=output / "host-stderr.log",
        )
        bridge = RoboDojoTools(connection, output=output / "artifacts")
        runtime = build_runtime(
            bridge, OpenAICompatiblePlannerBackend(provider), memory_root=output / "memory",
            context_window_tokens=provider.context_window_tokens,
        )
        runner = OpenEtaEpisodeRunner(runtime=runtime, environment=RoboDojoEpisodeEnvironment(bridge))
        result = runner.run(
            task=task, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
            timeout_s=args.timeout, max_total_tokens=args.max_total_tokens, metadata=run_metadata,
        )
        (output / "episode.json").write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        summary.update(
            status="failed" if bridge.failed else "completed",
            task_success=bridge.status.get("official_task_success") is True,
            outcome=bridge.status.get("outcome"), session_id=result.session_id,
            num_turns=len(result.steps), metadata=result.metadata,
        )
    except Exception as exc:
        # Keep credential-bearing SDK errors out of console and persisted logs.
        summary["error_type"] = type(exc).__name__
    finally:
        if bridge is not None:
            summary["cleanup"] = bridge.close()
        elif connection is not None:
            summary["cleanup"] = connection.close()
        if summary.get("cleanup", {}).get("ok") is False:
            summary["status"] = "failed"
            summary["cleanup_incomplete"] = True
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary.get(key) for key in ("status", "task_success", "error_type")}))
    print(f"Results: {output}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
