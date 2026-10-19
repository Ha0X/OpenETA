"""Connect OpenETA's runtime to the existing checked RoboDojo task Host.

The Host runs in its own worktree/interpreter and owns the simulator episode.
Only its public MCP schema/content crosses this boundary; no Codex CLI, model
client, private simulator packet, or in-process cross-worktree import is used.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import Future
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import re
import threading
import time
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.episode import ToolFeedbackEpisodeEnvironment
from agent.runtime.image_artifacts import materialize_mcp_images
from agent.runtime.response_artifacts import build_observation_snapshot, materialize_json_response
from agent.tools.registry import (
    ENVIRONMENT_AUTHORITY, ToolEffect, ToolExecutionContext, ToolRegistry, ToolSpec,
    make_tool_result,
)


REMOTE_TOOLS = (
    "episode_status", "observe", "mark_point", "read_skill", "move_to",
    "gripper_control", "wait", "check_task_completion", "finish_episode",
)
MUTATING_TOOLS = {"move_to", "gripper_control", "wait", "finish_episode"}
TOOL_NAMES = {name: name if name == "observe" else f"robodojo_{name}" for name in REMOTE_TOOLS}


class RoboDojoHostConnection:
    """One persistent stdio MCP session, with synchronous calls for the runtime.

    Unlike per-call stdio transports, this never restarts the task Host between
    actions. A timeout invalidates the connection; mutations are never retried.
    """

    def __init__(self, *, command: str, args: list[str], cwd: Path, stderr_path: Path,
                 startup_timeout_s: float = 600):
        self.command, self.args, self.cwd = command, args, cwd
        self.stderr_path = stderr_path
        self._ready: Future = Future()
        self._closed = False
        self._failure: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="robodojo-mcp", daemon=True)
        self._thread.start()
        try:
            self._ready.result(timeout=startup_timeout_s)
        except BaseException:
            self.close()
            raise

    def _run(self):
        try:
            asyncio.run(self._serve())
        except BaseException as exc:
            self._failure = exc
            if not self._ready.done():
                self._ready.set_exception(exc)

    async def _serve(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._stop = asyncio.Event()
        params = StdioServerParameters(command=self.command, args=self.args, cwd=str(self.cwd))
        self.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        with self.stderr_path.open("a", encoding="utf-8") as stderr:
            async with stdio_client(params, errlog=stderr) as (read, write):
                async with ClientSession(read, write) as session:
                    self._session = session
                    await session.initialize()
                    self._ready.set_result(True)
                    await self._stop.wait()

    def _request(self, operation, *, timeout_s: float):
        if self._closed or self._failure or not self._thread.is_alive():
            raise RuntimeError("RoboDojo MCP connection is closed or failed; do not retry actions")
        future = asyncio.run_coroutine_threadsafe(operation(), self._loop)
        try:
            return future.result(timeout=timeout_s)
        except BaseException:
            future.cancel()
            self._closed = True
            self._loop.call_soon_threadsafe(self._stop.set)
            raise

    def list_tools(self, *, timeout_s=30):
        result = self._request(self._session.list_tools, timeout_s=timeout_s)
        return result.model_dump(mode="json")

    def call_tool(self, name, arguments, *, timeout_s=450):
        result = self._request(
            lambda: self._session.call_tool(name, arguments,
                                           read_timeout_seconds=timedelta(seconds=timeout_s)),
            timeout_s=timeout_s + 5,
        )
        return result.model_dump(mode="json")

    def close(self):
        self._closed = True
        if self._thread.is_alive() and hasattr(self, "_loop"):
            if self._ready.done() and self._ready.exception() is None:
                self._loop.call_soon_threadsafe(self._stop.set)
            else:
                self._loop.call_soon_threadsafe(self._task.cancel)
            self._thread.join(timeout=100)
        return {"ok": not self._thread.is_alive(), "pending": self._thread.is_alive()}


class RoboDojoTools:
    """Adapt public Host responses into trusted OpenETA tool feedback."""

    def __init__(self, transport, *, output: Path):
        self.transport = transport
        self.output = output
        self.host_id = uuid4().hex
        self.status: JsonDict = {}
        self.failed = False
        self._snapshot: JsonDict | None = None
        self._snapshot_action_index: int | None = None
        self._lock = threading.RLock()
        self.registry = ToolRegistry()
        definitions = {tool["name"]: tool for tool in transport.list_tools()["tools"]}
        missing = set(REMOTE_TOOLS) - definitions.keys()
        if missing:
            raise ValueError(f"RoboDojo physical-contact tool profile is incomplete: {sorted(missing)}")
        for remote_name in REMOTE_TOOLS:
            definition = definitions[remote_name]
            schema = deepcopy(definition["inputSchema"])
            if remote_name == "observe":
                # The runtime may supply a host-owned refresh explanation.
                schema["properties"]["reason"] = {"type": "string"}
            name = TOOL_NAMES[remote_name]
            self.registry.register(ToolSpec(
                name=name,
                category="control" if remote_name in MUTATING_TOOLS else "perception",
                description=definition.get("description", "") + (
                    " Packaged skill text uses native names; call the corresponding "
                    "robodojo_ prefixed tool here (observe keeps its name)."
                ),
                parameters=schema.get("properties", {}),
                request_schema=schema,
                effect=ToolEffect.WORLD_MUTATING if remote_name in MUTATING_TOOLS else ToolEffect.READ_ONLY,
                batchable=False,
            ))
            self.registry.bind_handler(name, self._handler(remote_name), authority=ENVIRONMENT_AUTHORITY)

    def _handler(self, remote_name):
        def call(context: ToolExecutionContext):
            with self._lock:
                if self.failed:
                    return self._failure_result(context, "previous_execution_outcome_unknown")
                arguments = dict(context.parameters)
                if remote_name == "observe":
                    arguments.pop("reason", None)
                # Reject non-finite model-authored numbers before crossing MCP.
                json.dumps(arguments, allow_nan=False)
                try:
                    response = self.transport.call_tool(remote_name, arguments, timeout_s=450)
                    return self._result(context, remote_name, response)
                except Exception:
                    self.failed = True
                    return self._failure_result(context, "transport_or_execution_outcome_unknown")
        return call

    def _environment_receipt(self, context, **values):
        return {
            "schema_version": "openeta.environment_receipt.v1",
            "receipt_id": uuid4().hex, "backend": "robodojo_task_host",
            "agent_tool": context.name, "execution_id": context.metadata.get("execution_id", ""),
            "agent_session_id": context.metadata.get("session_id", ""),
            "simulator_session_id": self.host_id, "handle": self.host_id,
            "timestamp_s": time.time(), "reward_present": False,
            "observation_fresh": False, **values,
        }

    def _failure_result(self, context, code):
        return make_tool_result(
            context, success=False, content=code,
            diagnostics=[{"code": code, "automatic_retry_allowed": False}],
            environment_receipt=self._environment_receipt(context, truncated=True),
        )

    def _result(self, context, remote_name, response):
        blocks = response.get("content", [])
        if response.get("isError") or not blocks or blocks[0].get("type") != "text":
            raise ValueError("RoboDojo Host returned no structured public result")
        public = json.loads(blocks[0]["text"])
        status = public["status"]
        outcome = status["outcome"]
        official_success = outcome.get("status") == "success" and outcome.get("benchmark_success") is True
        if bool(status.get("official_task_success")) != official_success:
            raise ValueError("Inconsistent official outcome")
        self.status = status
        error = public.get("error") or status.get("error")
        unknown = error and error.get("code") in {
            "transport_or_execution_outcome_unknown", "evidence_write_failed_after_dispatch",
        }
        if unknown:
            self.failed = True
        index = public.get("action_index")
        originals, extra_images = [], []
        label = ""
        directory = {camera["name"]: camera for camera in public.get("cameras", [])}
        for block in blocks[1:]:
            if block.get("type") == "text":
                label = block["text"]
            elif block.get("type") == "image":
                match = re.match(r"(.+): original \d+×\d+ image,", label)
                camera = match.group(1) if match else f"{remote_name}_derived_{len(extra_images)}"
                row = {
                    "frame_id": camera, "role": "scene_primary",
                    "rgb_base64": block["data"],
                    "rgb_base64_format": block["mimeType"].split("/")[-1],
                    "label": label,
                }
                if match:
                    row.update({key: directory[camera][key] for key in ("width", "height")})
                    originals.append(row)
                else:
                    extra_images.append(row)
        # Materialize only public MCP images, including labelled overlays/crops.
        bundle = materialize_mcp_images(
            {"cameras": originals + extra_images}, output_root=self.output / "images",
        )
        artifacts = [image.to_dict() for image in bundle.images]
        for artifact, row in zip(artifacts, originals + extra_images):
            artifact["label"] = row["label"]
        evidence = public.get("evidence", {})
        received = evidence.get("after_observation_received") is True
        snapshot = None
        if not unknown and received and originals:
            observation = {
                "task": status["task"], "robot": public.get("robot", {}), "objects": [],
                "cameras": bundle.payload["cameras"][:len(originals)],
                "metadata": {"native_action_index": index, "observation_source": "public_host_mcp",
                             "observation_clock": public.get("observation_clock", {}),
                             "tcp_state": public.get("tcp_state", {}),
                             "pose_semantics": {
                                 "tcp_state": "Measured control TCP; xyz in world metres, approach=+Z, jaw=+X in world axes.",
                                 "robot_arms_end_effector_pose": "Raw link pose; not the calibrated TCP used by move_to. Do not substitute it for tcp_state.",
                             }},
            }
            snapshot = build_observation_snapshot({"observation": observation}, image_artifacts=artifacts)
            self._snapshot, self._snapshot_action_index = deepcopy(snapshot), index
        elif not unknown and self._snapshot is not None and index == self._snapshot_action_index:
            # Same native state: a read-only result or a rejected zero-step move.
            # Keep the original frame identities; do not relabel stale frames.
            receipt = public.get("receipt") or {}
            unchanged = remote_name not in MUTATING_TOOLS or (
                receipt.get("native_actions") == 0 and receipt.get("physics_steps") == 0
            )
            if unchanged:
                snapshot = deepcopy(self._snapshot)
                meta = snapshot["observation"]["metadata"]
                meta["observation_source"] = "cached_same_native_action_index"
                meta["image_artifacts"] = artifacts + meta.get("image_artifacts", [])
        response_artifact = materialize_json_response(
            public, output_root=self.output / "responses", name=remote_name,
        )
        # Reward derives exclusively from the Host's validated official outcome.
        # A locally closed/failed attempt is truncation, not benchmark termination.
        receipt = self._environment_receipt(
            context, remote_tool=remote_name,
            reward_present=outcome.get("status") in {"running", "success", "failure"},
            reward=1.0 if official_success else 0.0,
            terminated=outcome.get("terminal") is True,
            truncated=bool(unknown or (status.get("closed") and not outcome.get("terminal"))),
            observation_fresh=snapshot is not None,
        )
        if snapshot:
            receipt["observation_snapshot"] = snapshot
        outputs = {**public, "schema_version": "openeta.robodojo_public_result.v1",
                   "response_path": response_artifact.path}
        return make_tool_result(
            context, success=not bool(error),
            content=json.dumps({"status": status, "error": error}, ensure_ascii=False),
            outputs=outputs, artifacts=[*artifacts, response_artifact.to_dict()],
            environment_receipt=receipt,
            diagnostics=[error] if error else [],
        )

    def close(self):
        with self._lock:
            cleanup = self.status.get("cleanup")
            if not self.failed and not self.status.get("closed"):
                try:
                    result = self.transport.call_tool(
                        "finish_episode", {"success": False, "reason": "openeta_run_finished"},
                        timeout_s=120,
                    )
                    self.status = json.loads(result["content"][0]["text"])["status"]
                    cleanup = self.status.get("cleanup")
                except Exception:
                    self.failed = True
            connection = self.transport.close()
            return {"ok": bool(cleanup and cleanup.get("ok") and connection.get("ok")),
                    "host_cleanup": cleanup, "connection": connection}


class RoboDojoEpisodeEnvironment(ToolFeedbackEpisodeEnvironment):
    """Use normal trusted-tool feedback and close the owned task Host on exit."""

    def __init__(self, bridge: RoboDojoTools):
        super().__init__()
        self.bridge = bridge

    def close(self):
        return self.bridge.close()
