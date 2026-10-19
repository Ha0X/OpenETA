from adapter.protocol import RobotState
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import PlannerDecision, _validate_planner_decision, build_tool_context
from agent.runtime.skills import SkillRegistry
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.episode import ToolFeedbackEpisodeEnvironment
from agent.runtime.response_artifacts import build_observation_summary
from agent.tools.registry import ToolEffect, ToolRegistry, ToolSpec, build_default_tool_registry


def _tools():
    calls = []
    tools = ToolRegistry()
    tools.register(ToolSpec(
        name="robodojo_gripper_control", category="control",
        description="Choose an arm and continuous gripper opening.",
        effect=ToolEffect.WORLD_MUTATING,
        request_schema={
            "type": "object", "additionalProperties": False,
            "required": ["role", "openness"],
            "properties": {
                "role": {"type": "string", "enum": ["left", "right"]},
                "openness": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    ), lambda context: calls.append(context.parameters))
    return tools, calls


def test_explicit_schema_reaches_model_and_rejects_before_dispatch():
    tools, calls = _tools()
    observation = ToolFeedbackEpisodeEnvironment().reset(task="pick cup")
    memory = AgentMemory()
    memory.start_session(task="pick cup")
    context = build_tool_context(
        observation=observation, memory=memory, tools=tools, skills=SkillRegistry(),
    )
    schema = context["agent_context"]["available_tools"][0]["parameters"]
    assert schema == tools.get("robodojo_gripper_control").request_schema
    for parameters in ({"role": "left", "openness": 2},
                       {"role": "middle", "openness": .65},
                       {"role": "right", "openness": True},
                       {"role": "right", "position": 1}):
        decision = PlannerDecision(action_type="tool_call", action="robodojo_gripper_control",
                                   parameters=parameters)
        assert _validate_planner_decision(decision, tools, SkillRegistry())
        assert not tools.call(decision.action, parameters).success
    assert calls == []
    tools.call("robodojo_gripper_control", {"role": "right", "openness": .65})
    assert calls == [{"role": "right", "openness": .65}]


def test_dual_arm_state_survives_roundtrip_and_summary():
    payload = {"arms": {
        "left": {"joint_names": ["joint1"], "joint_positions": [1.0],
                 "end_effector_pose": {"xyz": [1, 2, 3]}},
        "right": {"joint_names": ["joint1"], "joint_positions": [2.0],
                  "gripper_state": {"openness": .65, "command": .7}},
    }}
    state = RobotState.from_dict(payload)
    assert state.arms["left"].joint_positions == [1.0]
    restored = RobotState.from_dict(state.to_dict())
    assert restored.arms["right"].gripper_state == payload["arms"]["right"]["gripper_state"]
    summary = build_observation_summary({"observation": {"robot": state.to_dict()}})
    assert summary["robot"]["arms"]["right"]["gripper_state"]["openness"] == .65
    assert "arms" not in RobotState().to_dict()


def test_legacy_motion_and_gripper_contracts_remain_unchanged():
    tools = build_default_tool_registry()
    for name in ("move_to", "gripper_control"):
        tools.bind_handler(name, lambda context: None)
    assert _validate_planner_decision(
        PlannerDecision(action_type="tool_call", action="gripper_control",
                        parameters={"position": .65}), tools, SkillRegistry(),
    )
    assert _validate_planner_decision(
        PlannerDecision(action_type="tool_call", action="move_to",
                        parameters={"role": "right", "target": {"xyz": [0, 0, 1]}}),
        tools, SkillRegistry(),
    )


def test_runtime_accepts_explicit_backend_tool_set():
    tools, _ = _tools()
    runtime = OpenEtaAgentRuntime(tools=tools, skills=SkillRegistry())
    assert [spec.name for spec in runtime.tools.list()] == ["robodojo_gripper_control"]
