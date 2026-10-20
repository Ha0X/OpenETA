import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from scripts.robodojo_endpoint_diagnostic import make_probes, run_probes, validate_plan


def test_probes_change_one_factor_and_preserve_logged_targets(tmp_path):
    def convert(value, current):
        result = current.copy()
        result[:3, 3] = value["xyz"]
        return result.tolist()

    for n, reason in ((24, "ik_search_no_valid_solution"), (25, "ik_candidates_in_collision"),
                      (27, "ik_search_no_valid_solution"), (28, "ik_candidates_in_collision"),
                      (29, "ik_search_no_valid_solution")):
        directory = tmp_path / f"host/calls/{n:06d}"
        directory.mkdir(parents=True)
        (directory / "request.json").write_text(json.dumps({
            "tool": "move_to", "arguments": {"role": "right", "target": {"xyz": [n / 100, .2, .8]}}}))
        (directory / "response.json").write_text(json.dumps({"content": [{"type": "text", "text": json.dumps({
            "receipt": {"stop": {"phase": "endpoint_preview", "reason": reason}}})}]}))
    current = np.eye(4)
    current[:3, 3] = [.1, .2, .9]
    packet = {"action_index": 128, "control": {"tcp_poses": {"right": current.tolist()}}}
    plan = make_probes(tmp_path, 23, packet, convert)
    validate_plan(plan)
    assert len(plan["probes"]) == 20
    assert np.array_equal(plan["probes"][0]["target"], current)
    assert [p["source_call"] for p in plan["probes"][1:6]] == ["000024", "000025", "000027", "000028", "000029"]
    anchor = np.asarray(plan["probes"][2]["target"])
    for probe in plan["probes"][8:]:
        target = np.asarray(probe["target"])
        assert probe["source_call"] == "000025"
        if probe["change"] == "position":
            assert np.array_equal(target[:3, :3], anchor[:3, :3])
            assert np.isclose(np.linalg.norm(target[:3, 3] - anchor[:3, 3]), .02)
        else:
            assert np.array_equal(target[:3, 3], anchor[:3, 3])
            assert np.isclose(np.arccos((np.trace(target[:3, :3]) - 1) / 2), np.pi / 12)
    assert packet["control"]["tcp_poses"]["right"] == current.tolist()


def test_limits_and_invalid_transforms_rejected():
    plan = {"total_timeout_s": 600, "per_probe_timeout_s": 30,
            "probes": [{"id": str(i), "target": np.eye(4).tolist()} for i in range(21)]}
    with pytest.raises(ValueError, match="1–20"):
        validate_plan(plan)
    plan["probes"] = plan["probes"][:1]
    plan["per_probe_timeout_s"] = 31
    with pytest.raises(ValueError, match="budget"):
        validate_plan(plan)
    plan["per_probe_timeout_s"] = 30
    plan["probes"][0]["target"][0][0] = 2
    with pytest.raises(ValueError, match="rigid transform"):
        validate_plan(plan)


@pytest.mark.parametrize("change_state", [False, True])
def test_probe_is_read_only_and_aborts_on_state_change(tmp_path, monkeypatch, change_state):
    class Control:
        roles = ["right"]
        contact_grants = {}
        allowed_pairs = []

        def _action(self):
            raise AssertionError("Original action must never run")

        def tcp(self, index):
            return np.eye(4)

        def _contact_scene(self, scene):
            return scene

        def _public_check(self, detail):
            return detail

        def _public_plan(self, detail):
            return detail

    scene = SimpleNamespace(check=lambda *args, **kwargs: {"clear": True})
    control = Control()
    control.world = SimpleNamespace(refresh=lambda: None, positions=lambda: [np.zeros(6)])
    adapter = SimpleNamespace(
        _step_count=128, _state_valid=True, _checked_controller=control,
        _observation_clock={"counter": 123},
        _env=SimpleNamespace(take_action_cnt=[128], sim=SimpleNamespace(_sim_step_counter=123)),
    )
    attached = ModuleType("sim.envs.robodojo.attached_geometry")
    attached.attached_scene = lambda *args: (scene, {})
    planning = ModuleType("sim.envs.robodojo.motion_planning")
    calls = []

    def combined(*args):
        with pytest.raises(RuntimeError, match="forbidden"):
            control._action()
        calls.append(True)
        if change_state:
            adapter._env.sim._sim_step_counter += 1
        return [], {"reason": "ik_search_no_valid_solution"}

    planning.combined_goals = combined
    monkeypatch.setitem(sys.modules, attached.__name__, attached)
    monkeypatch.setitem(sys.modules, planning.__name__, planning)
    plan = {"total_timeout_s": 600, "per_probe_timeout_s": 30,
            "scope": "endpoint_only", "expected_action_index": 128,
            "probes": [{"id": str(i), "role": "right", "target": np.eye(4).tolist()} for i in range(2)]}
    if change_state:
        with pytest.raises(RuntimeError, match="State changed"):
            run_probes(adapter, plan, tmp_path)
    else:
        run_probes(adapter, plan, tmp_path)
    report = json.loads((tmp_path / "endpoint-probes.json").read_text())
    assert report["state_unchanged"] is not change_state
    assert report["physics_advanced"] is change_state
    assert len(calls) == (1 if change_state else 2)
    assert "_action" not in control.__dict__
