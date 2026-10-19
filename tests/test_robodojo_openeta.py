from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from agent.backends.planner import PlannerBackendRequest, StaticPlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.planner import build_tool_context
from agent.tools.robodojo import (
    REMOTE_TOOLS, RoboDojoEpisodeEnvironment, RoboDojoHostConnection, RoboDojoTools,
)
from scripts.robodojo_openeta_run import (
    build_backend, build_runtime, host_command, load_reference_guidance, parser,
    reference_instructions,
)


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


def test_model_input_preserves_control_frames_skill_text_and_motion_feedback(tmp_path):
    host = FakeHost()
    call = host.call_tool
    tcp = {role: {"xyz": [x, -.2073, .9215], "approach": [0, 1, 0], "jaw": [-1, 0, 0]}
           for role, x in (("left", -.3), ("right", .3))}
    gripper = {"commanded_openness": .2, "measurement_available": True,
               "aperture": {"width_m": .027, "width_limits_m": [.001, .089]}}
    skill_text = "WORKFLOW_START\n" + "Check measured co-motion before transport.\n" * 140 + "WORKFLOW_END"

    def public_result(name, arguments, **kwargs):
        result = call(name, arguments, **kwargs)
        public = json.loads(result["content"][0]["text"])
        public["tcp_state"] = deepcopy(tcp)
        public["robot"]["arms"]["right"].update(
            end_effector_pose={"xyz": [.3, -.3523, .9215]}, gripper_state=gripper,
        )
        # Real Host responses exceed the generic 8 KB output threshold.
        public["robot"]["metadata"] = {"diagnostics": "x" * 9000}
        if name == "read_skill":
            public["receipt"] = {"skill_id": "pick-place", "documents": [
                {"reference": "pick-place.md", "text": skill_text},
            ]}
        elif name == "move_to":
            public["receipt"] = {
                **{f"diagnostic_{i}": i for i in range(20)},
                "reached_target": False, "native_actions": 1, "physics_steps": 10,
                "final_tcp_error": {"right": {"position_m": .012}},
                "stop": {"reason": "physical_contact_penetration_limit", "contacts": [
                    {"participants": [{"kind": "robot", "role": "right", "part": "finger"},
                                      {"kind": "object", "part": "surface"}]},
                ]},
            }
            public["motion_feedback"] = {"guidance": ["Contact is not grasp confirmation."]}
        result["content"][0]["text"] = json.dumps(public)
        return result

    host.call_tool = public_result
    _, backend, _, _, _ = run_fixture(tmp_path, host, [
        decision("observe"), decision("robodojo_read_skill"),
        decision("robodojo_move_to", role="right", target={"xyz": [.2, 0, .8]}),
        {"kind": "response", "name": "talk", "parameters": {"message": "done checking"}},
    ])
    contexts = [r.tool_context for r in backend.requests]
    observation = contexts[1]["current_observation"]["summary"]
    assert observation["metadata"]["tcp_state"] == tcp
    assert observation["robot"]["arms"]["right"]["end_effector_pose"]["xyz"] == [.3, -.3523, .9215]
    assert "Raw link pose" in observation["metadata"]["pose_semantics"]["robot_arms_end_effector_pose"]
    assert observation["gripper_evidence"]["by_role"]["right"] == gripper
    assert "binary latch" not in observation["gripper_evidence"]["semantics"]
    skill = contexts[2]["decision_state"]["last_action_effect"]["outputs"]
    assert skill["receipt"]["documents"][0]["text"] == skill_text
    motion = contexts[3]["decision_state"]["last_action_effect"]["outputs"]
    assert motion["receipt"]["reached_target"] is False
    assert motion["receipt"]["physics_steps"] == 10
    assert motion["receipt"]["stop"]["contacts"][0]["participants"][0]["role"] == "right"
    assert motion["receipt"]["final_tcp_error"]["right"]["position_m"] == .012
    assert motion["motion_feedback"]["guidance"] == ["Contact is not grasp confirmation."]
    assert motion["tcp_state"] == tcp

    # Check the actual provider request, not just persisted runtime artifacts.
    api = build_backend(PlannerProviderConfig(model="test", api_base="https://example.test", api_key="test"))
    bodies = []

    def transport(url, body, headers, timeout_s):
        bodies.append(body)
        return {"choices": [{"message": {"content": "<decision><kind>response</kind><name>talk</name><parameters/></decision>"}}]}

    api.transport = transport
    api.decide(backend.requests[2])
    sent = json.dumps(bodies[0])
    assert "WORKFLOW_START" in sent and "WORKFLOW_END" in sent
    assert "-0.2073" in sent and "-0.3523" in sent


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


def test_reference_workflow_and_rules_reach_provider_without_codex(tmp_path):
    root = tmp_path / "reference"
    (root / "tools").mkdir(parents=True)
    (root / "tools/__init__.py").write_text("")
    (root / "tools/harness_config.py").write_text('''
class ConfigRegistry:
    def resolve(self, config_id):
        assert config_id == "preset_v1"
        return type("Config", (), {"sha256": "bound-hash", "value": {
            "components": {"benchmark": {"completion_feedback": "native_coarse"}}
        }})()
''')
    (root / "tools/task_rule_supplements.py").write_text('''
def task_rule_supplement(benchmark, env_id):
    if env_id != "matching-env":
        raise ValueError("Task rule does not match environment")
    return "\\nRestore the cup upright.", {"id": "public_rule_v1"}
''')
    skill = root / "plugins/openeta/skills/openeta-robodojo/SKILL.md"
    skill.parent.mkdir(parents=True)
    source = "---\nname: test\n---\nUse waypoints and read_skill as needed.\n"
    skill.write_text(source)
    guidance = load_reference_guidance(root=root, python=sys.executable,
                                       config_id="preset_v1", env_id="matching-env")
    assert guidance["workflow_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert guidance["config_sha256"] == "bound-hash"
    instructions = reference_instructions(guidance, max_turns=80, max_tool_calls=82, timeout=1200)
    backend = build_backend(PlannerProviderConfig(model="test", api_base="https://example.test", api_key="test"))
    bodies = []

    def transport(url, body, headers, timeout_s):
        bodies.append(body)
        return {"choices": [{"message": {"content": "<decision><kind>tool_call</kind><name>observe</name><parameters/></decision>"}}]}

    backend.transport = transport
    bridge = RoboDojoTools(FakeHost(), output=tmp_path / "artifacts")
    runtime = build_runtime(bridge, backend, memory_root=tmp_path / "memory", task_guidance=instructions)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=RoboDojoEpisodeEnvironment(bridge))
    runner.run(task="pour balls", max_turns=1, timeout_s=30)
    sent = bodies[0]["messages"][0]["content"]
    assert guidance["workflow"] in sent
    assert "Restore the cup upright." in sent
    assert "82 Host requests, 1200 seconds" in sent
    assert "oracle-assisted" in sent and "robodojo_ tools" in sent
    assert "name: test" not in sent
    with pytest.raises(subprocess.CalledProcessError):
        load_reference_guidance(root=root, python=sys.executable,
                                config_id="preset_v1", env_id="wrong-env")


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


@pytest.mark.parametrize("effort", [None, "none", "high"])
def test_explicit_reasoning_effort_reaches_api_without_sampling_conflict(effort):
    backend = build_backend(PlannerProviderConfig(
        model="gpt-6-astra", api_base="https://example.test", api_key="test", max_attempts=1,
    ), reasoning_effort=effort)
    bodies = []

    def transport(url, body, headers, timeout_s):
        bodies.append(body)
        return {"choices": [{"message": {"content": "<decision><kind>tool_call</kind><name>observe</name><parameters/></decision>"}}]}

    backend.transport = transport
    backend.decide(PlannerBackendRequest(tool_context={"task": "test"}))
    body = bodies[0]
    assert body["model"] == "gpt-6-astra"
    if effort is None:
        assert "reasoning_effort" not in body
        assert body["max_tokens"] == 8192 and body["temperature"] == 0.0
    else:
        assert body["reasoning_effort"] == effort
        assert body["max_completion_tokens"] == 8192 and "max_tokens" not in body
        assert ("temperature" in body) == (effort == "none")
    assert backend.config.redacted()["reasoning_effort"] == effort
