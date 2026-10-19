from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
from pathlib import Path
import sys

from PIL import Image
import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.planner import build_tool_context
from agent.tools.robodojo import (
    REMOTE_TOOLS, RoboDojoEpisodeEnvironment, RoboDojoHostConnection, RoboDojoTools,
)
from scripts.robodojo_openeta_run import build_backend, build_runtime, host_command, parser


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required),
            "additionalProperties": False}


def definitions():
    role = {"type": "string", "enum": ["left", "right"]}
    vector = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    target = obj({"xyz": vector, "delta_xyz": vector, "frame": {"enum": ["world", "tcp"]}})
    target["oneOf"] = [{"required": ["xyz"]}, {"required": ["delta_xyz"]}]
    motion = obj({"role": role, "target": target}, ["role", "target"])
    schemas = {name: obj() for name in REMOTE_TOOLS}
    schemas["move_to"] = {"type": "object", "oneOf": [motion, obj({
        "motions": {"type": "array", "items": motion, "minItems": 1, "maxItems": 2},
        "coordination": {"enum": ["independent", "synchronized"]},
    }, ["motions"])]}
    schemas["gripper_control"] = obj({
        "role": role, "openness": {"type": "number", "minimum": 0, "maximum": 1},
    }, ["role", "openness"])
    schemas["finish_episode"] = obj({"success": {"type": "boolean"}, "reason": {"type": "string"}},
                                    ["success", "reason"])
    schemas["observe"] = obj({"cameras": {"type": "array", "items": {"type": "string"}}})
    return {"tools": [{"name": name, "description": name, "inputSchema": schema}
                      for name, schema in schemas.items()]}


def image_block():
    data = io.BytesIO()
    Image.new("RGB", (4, 4)).save(data, format="WEBP", lossless=True)
    return {"type": "image", "mimeType": "image/webp", "data": base64.b64encode(data.getvalue()).decode()}


class FakeHost:
    def __init__(self, *, success_on_move=False, timeout_on_move=False, omit_motion_images=False):
        self.calls = []
        self.index = 0
        self.success_on_move = success_on_move
        self.timeout_on_move = timeout_on_move
        self.omit_motion_images = omit_motion_images
        self.closed = False

    def list_tools(self):
        return definitions()

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append((name, deepcopy(arguments)))
        if name == "move_to" and self.timeout_on_move:
            raise TimeoutError("Outcome unknown")
        mutation = name in {"move_to", "gripper_control", "wait"}
        if mutation:
            self.index += 1
        self.closed = name == "finish_episode"
        success = self.success_on_move and self.index > 0
        status = {
            "task": "pick cup", "closed": self.closed, "official_task_success": success,
            "outcome": {"status": "success" if success else "running", "terminal": success,
                        "benchmark_success": True if success else None, "action_index": self.index},
            "cleanup": {"ok": True} if self.closed else None, "error": None,
        }
        public = {
            "status": status, "action_index": self.index, "error": None,
            "evidence": {"after_observation_received": name != "read_skill"},
            "robot": {"arms": {
                "left": {"end_effector_pose": {"xyz": [.1, .2, .3]}},
                "right": {"gripper_state": {"openness": .65, "command": .7}},
            }},
            "receipt": {"native_actions": 1 if mutation else 0,
                        "physics_steps": 10 if mutation else 0, "reached_target": False},
            "cameras": [{"name": name, "width": 4, "height": 4}
                        for name in ("head", "left_wrist", "right_wrist")],
        }
        content = [{"type": "text", "text": json.dumps(public)}]
        if name != "read_skill" and not (mutation and self.omit_motion_images):
            for camera in public["cameras"]:
                content.extend([{"type": "text", "text": f'{camera["name"]}: original 4×4 image, action_index={self.index}'}, image_block()])
        return {"content": content, "isError": False}

    def close(self):
        return {"ok": True}


class CapturingBackend(StaticPlannerBackend):
    def __init__(self, decisions):
        super().__init__(decisions)
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        return super().decide(request)


def decision(name, **parameters):
    return {"kind": "tool_call", "name": name, "parameters": parameters}


def run_fixture(tmp_path, host, decisions):
    bridge = RoboDojoTools(host, output=tmp_path / "artifacts")
    backend = CapturingBackend(decisions)
    runtime = build_runtime(bridge, backend, memory_root=tmp_path / "memory")
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=RoboDojoEpisodeEnvironment(bridge))
    result = runner.run(task="pick cup", max_turns=len(decisions), timeout_s=30,
                        metadata={"require_official_reward": True})
    return bridge, backend, runtime, runner, result


def test_main_runtime_executes_dual_motion_and_continuous_gripper(tmp_path):
    motions = [{"role": role, "target": {"delta_xyz": [0, 0, .02], "frame": "world"}}
               for role in ("left", "right")]
    host = FakeHost()
    bridge, backend, runtime, runner, result = run_fixture(tmp_path, host, [
        decision("observe"),
        decision("robodojo_move_to", motions=motions, coordination="synchronized"),
        decision("robodojo_gripper_control", role="right", openness=.65),
        decision("robodojo_finish_episode", success=False, reason="test_complete"),
    ])
    assert host.calls[1] == ("move_to", {"motions": motions, "coordination": "synchronized"})
    assert host.calls[2] == ("gripper_control", {"role": "right", "openness": .65})
    assert result.truncated and not result.terminated
    assert bridge.status["official_task_success"] is False
    assert runner.current_observation.robot.arms["right"].gripper_state["openness"] == .65
    request = backend.requests[1].tool_context
    serialized = json.dumps(request)
    assert "right_wrist" in serialized and "left_wrist" in serialized and "head" in serialized
    context = build_tool_context(observation=runner.current_observation, memory=runtime.memory,
                                 tools=runtime.tools, skills=runtime.skills,
                                 config=runtime.planner.context_config)
    assert len(context["vision_image_paths"]) == 3
    assert all(Path(path).suffix == ".webp" for path in context["vision_image_paths"])
    assert context["observation"]["robot"]["arms"]["right"]["gripper_state"]["openness"] == .65
    assert "rgb_base64" not in serialized
    assert list((tmp_path / "memory" / "sessions").glob("*/trace.jsonl"))
    assert bridge.close()["ok"] is True


def test_only_official_outcome_establishes_success(tmp_path):
    host = FakeHost(success_on_move=True)
    bridge, _, _, _, result = run_fixture(tmp_path, host, [
        decision("observe"), decision("robodojo_move_to", role="right", target={"xyz": [.2, 0, .8]}),
    ])
    assert result.terminated and not result.truncated
    assert bridge.status["official_task_success"] is True
    assert result.steps[-1].step_result.reward == 1.0
    assert bridge.close()["ok"]


def test_timeout_stops_episode_without_retry_or_success(tmp_path):
    host = FakeHost(timeout_on_move=True)
    bridge, _, _, _, result = run_fixture(tmp_path, host, [
        decision("observe"), decision("robodojo_move_to", role="right", target={"xyz": [.2, 0, .8]}),
        decision("robodojo_gripper_control", role="right", openness=.65),
    ])
    assert bridge.failed and result.truncated
    assert [name for name, _ in host.calls] == ["observe", "move_to"]
    assert bridge.status["official_task_success"] is False


def test_missing_motion_images_forces_observe_before_next_model_call(tmp_path):
    host = FakeHost(omit_motion_images=True)
    _, _, _, _, _ = run_fixture(tmp_path, host, [
        decision("observe"), decision("robodojo_move_to", role="right", target={"xyz": [.2, 0, .8]}),
        decision("robodojo_gripper_control", role="right", openness=.65),
        decision("robodojo_finish_episode", success=False, reason="test_complete"),
    ])
    assert [name for name, _ in host.calls][:3] == ["observe", "move_to", "observe"]


@pytest.mark.parametrize("completion", [
    {"kind": "response", "name": "task_complete", "parameters": {}},
    {"kind": "response", "name": "talk", "parameters": {"success": True}},
])
def test_model_cannot_claim_completion_without_official_reward(tmp_path, completion):
    host = FakeHost()
    _, _, _, _, result = run_fixture(tmp_path, host, [
        decision("observe"), completion,
    ])
    # A validation-failure status report can end the local loop, but must not
    # become an agent task_complete command or an official success.
    request = result.steps[-1].action.command["request"]
    assert request["name"] == "talk"
    assert request["parameters"]["code"] == "planner_validation_failed"
    assert result.steps[-1].step_result.reward == 0
    assert host.index == 0


def test_observe_retains_backend_camera_schema(tmp_path):
    bridge = RoboDojoTools(FakeHost(), output=tmp_path)
    runtime = build_runtime(bridge, CapturingBackend([]), memory_root=tmp_path / "memory")
    assert "cameras" in runtime.planner.tool_contract_catalog.get("observe").request_schema["properties"]
    bad = bridge.registry.call("robodojo_move_to", {"role": "right", "target": {"xyz": [0, 0, 0], "delta_xyz": [0, 0, 0]}})
    assert not bad.success
    assert bridge.transport.calls == []


def test_mark_crop_and_cached_overlay_keep_distinct_image_identities(tmp_path):
    host = FakeHost()
    call = host.call_tool

    def with_derived_image(name, arguments, **kwargs):
        result = call(name, arguments, **kwargs)
        if name == "mark_point":
            result["content"] = result["content"][:1]
        result["content"].extend([
            {"type": "text", "text": "public crop" if name == "mark_point" else "TCP overlay"},
            image_block(),
        ])
        return result

    host.call_tool = with_derived_image
    bridge = RoboDojoTools(host, output=tmp_path)
    assert bridge.registry.call("observe", {}).success
    result = bridge.registry.call("robodojo_mark_point", {})
    assert result.success
    observation = result.details["environment_receipt"]["observation_snapshot"]["observation"]
    assert len(observation["cameras"]) == 3
    images = observation["metadata"]["image_artifacts"]
    assert len(images) == len({image["frame_id"] for image in images}) == 5
    assert images[0]["label"] == "public crop"


def test_persistent_stdio_session_keeps_same_host(tmp_path):
    script = tmp_path / "mcp_fixture.py"
    script.write_text('''
from mcp.server.fastmcp import FastMCP
server = FastMCP("counter")
count = 0
@server.tool()
def counter() -> int:
    global count
    count += 1
    return count
server.run(transport="stdio")
''')
    transport = RoboDojoHostConnection(command=sys.executable, args=[str(script)], cwd=tmp_path,
                                      stderr_path=tmp_path / "stderr.log", startup_timeout_s=30)
    try:
        assert transport.list_tools()["tools"][0]["name"] == "counter"
        first = transport.call_tool("counter", {})
        second = transport.call_tool("counter", {})
        assert first["content"][0]["text"] == "1"
        assert second["content"][0]["text"] == "2"
    finally:
        assert transport.close()["ok"]


def test_entry_point_starts_only_task_host(tmp_path):
    python_entry = tmp_path / ".venv/bin/python"
    python_entry.parent.mkdir(parents=True)
    python_entry.symlink_to(sys.executable)
    args = parser().parse_args([
        "--robodojo-root", str(tmp_path), "--sim-url", "http://node:123/mcp",
        "--config-id", "preset", "--env-id", "env", "--task-file", "task.txt",
        "--output", str(tmp_path / "run"),
    ])
    command, argv = host_command(args, root=tmp_path, output=tmp_path / "run", task="pick cup")
    assert command.endswith(".venv/bin/python")
    assert Path(command) == python_entry
    assert Path(command).resolve() != Path(command)
    assert argv[:2] == ["-m", "tools.codex_mcp_server"]
    assert "--expected-controller" in argv and "robodojo_checked" in argv
    assert "--model" not in argv and "--api-key" not in argv


def test_smoke_instructions_keep_checks_and_disable_validation_retries(tmp_path):
    bridge = RoboDojoTools(FakeHost(), output=tmp_path / "artifacts")
    runtime = build_runtime(bridge, CapturingBackend([]), memory_root=tmp_path / "memory",
                            smoke_test=True)
    assert "short live connectivity test" in runtime.planner.system_prompt
    assert "max_duration_s=0.5" in runtime.planner.system_prompt
    assert "Embodied Closed-Loop Operating Contract" in runtime.planner.system_prompt
    assert runtime.planner.max_validation_retries == 0


def test_api_backend_converts_provider_and_attaches_all_camera_views(tmp_path, monkeypatch):
    monkeypatch.setattr("agent.backends.planner.mimetypes.guess_type", lambda path: (None, None))
    provider = PlannerProviderConfig(model="test-model", api_base="https://example.test",
                                     api_key="test-key", timeout_s=60, max_attempts=1)
    backend = build_backend(provider)
    requests = []

    def transport(url, body, headers, timeout_s):
        assert timeout_s == 60
        requests.append(body)
        payload = (
            "<decision><kind>tool_call</kind><name>observe</name><parameters/></decision>"
            if len(requests) == 1 else
            "<decision><kind>response</kind><name>talk</name>"
            "<parameters><message>Observed</message></parameters></decision>"
        )
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": payload,
        }}], "usage": {"total_tokens": 10}}

    backend.transport = transport
    bridge = RoboDojoTools(FakeHost(), output=tmp_path / "artifacts")
    runtime = build_runtime(bridge, backend, memory_root=tmp_path / "memory", smoke_test=True)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=RoboDojoEpisodeEnvironment(bridge))
    runner.run(task="test", max_turns=2, timeout_s=30)
    assert len(requests) == 2
    images = [part for part in requests[1]["messages"][-1]["content"]
              if part["type"] == "image_url"]
    assert len(images) == 3
    assert backend.config.max_vision_images == 8
    assert backend.config.max_attempts == 1
