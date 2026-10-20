"""Operator-only, zero-step endpoint diagnosis using the unchanged RoboDojo solver.

This is not an agent tool or a policy. A recorded prefix must first be replayed
through the normal harness/Host. ``prepare`` writes at most 20 explicit endpoint
checks; ``worker`` wraps the reference worker and runs them on its next status
request. Results stay in operator files, outside the model-visible response.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def public_response(directory):
    response = json.loads((Path(directory) / "response.json").read_text())
    return json.loads(next(b["text"] for b in response["content"] if b["type"] == "text"))


def make_probes(baseline, after_call, packet, target_matrix):
    """Five logged failures, a measured-pose control, two repeats, twelve perturbations."""
    import numpy as np

    failures = []
    for directory in sorted((Path(baseline) / "host/calls").glob("[0-9]*")):
        if int(directory.name) <= after_call:
            continue
        request = json.loads((directory / "request.json").read_text())
        result = public_response(directory)
        receipt = result.get("receipt") or {}
        stop = receipt.get("stop") or {}
        if request["tool"] == "move_to" and stop.get("phase") == "endpoint_preview":
            arguments = request["arguments"]
            if arguments.get("waypoints") or "motions" in arguments:
                raise ValueError("This diagnostic requires single endpoints")
            failures.append((directory.name, arguments, stop["reason"]))
            if len(failures) == 5:
                break
    if not failures:
        raise ValueError("No endpoint failures after the replay checkpoint")
    role = failures[0][1]["role"]
    if any(arguments["role"] != role for _, arguments, _ in failures):
        raise ValueError("Use one holding arm per diagnostic")
    current = np.asarray(packet["control"]["tcp_poses"][role], dtype=float)
    probes = [{"id": "measured_pose_control", "role": role, "target": current.tolist()}]
    originals = []
    for call, arguments, reason in failures:
        probe = {
            "id": f"original_{call}", "role": role, "source_call": call,
            "baseline_reason": reason,
            "target": target_matrix(arguments["target"], current),
        }
        probes.append(probe)
        originals.append(probe)
    # Select by failure category, not by a hard-coded task coordinate.
    anchor = next((p for p in originals if p["baseline_reason"] == "ik_candidates_in_collision"),
                  originals[0])
    for index, original in enumerate((originals[0], anchor), 1):
        probes.append({**deepcopy(original), "id": original["id"] + f"_repeat_{index}"})
    for kind in ("position", "orientation"):
        for axis in range(3):
            for sign in (-1, 1):
                pose = np.asarray(anchor["target"], dtype=float).copy()
                if kind == "position":
                    pose[axis, 3] += sign * .02
                else:
                    # Rotate world-frame axes by 15 degrees; keep the TCP position.
                    theta = sign * np.pi / 12
                    vector = np.eye(3)[axis]
                    x, y, z = vector
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                    rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
                    pose[:3, :3] = rotation @ pose[:3, :3]
                probes.append({
                    "id": f"{kind}_{axis}_{sign:+d}", "role": role,
                    "source_call": anchor["source_call"], "change": kind,
                    "world_axis": axis, "signed_offset": sign * (.02 if kind == "position" else 15),
                    "offset_unit": "metres" if kind == "position" else "degrees",
                    "target": pose.tolist(),
                })
    return {"schema_version": "openeta.operator_endpoint_diagnostic.v1",
            "expected_action_index": packet["action_index"], "after_call": after_call,
            "max_probes": 20, "total_timeout_s": 600, "per_probe_timeout_s": 30,
            "scope": "endpoint_only_not_path_or_held_object_safety", "probes": probes}


def validate_plan(plan):
    import numpy as np

    probes = plan["probes"]
    if not 1 <= len(probes) <= 20 or len({p["id"] for p in probes}) != len(probes):
        raise ValueError("Expected 1–20 uniquely named probes")
    if not 0 < plan["total_timeout_s"] <= 600 or not 0 < plan["per_probe_timeout_s"] <= 30:
        raise ValueError("Probe time budget exceeds the diagnostic limit")
    for probe in probes:
        pose = np.asarray(probe["target"], dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
            raise ValueError("Probe target must be a rigid transform")


def state_signature(adapter, control):
    return {
        "action_index": int(adapter._step_count),
        "native_action_count": int(adapter._env.take_action_cnt[0]),
        "physics_step_counter": int(adapter._env.sim._sim_step_counter),
        "joints": [q.tolist() for q in control.world.positions()],
        "tcp": {role: control.tcp(i).tolist() for i, role in enumerate(control.roles)},
        "observation_clock": deepcopy(adapter._observation_clock),
    }


def run_probes(adapter, plan, output):
    import numpy as np
    from sim.envs.robodojo.attached_geometry import attached_scene
    from sim.envs.robodojo.motion_planning import combined_goals

    validate_plan(plan)
    if adapter._step_count != plan["expected_action_index"] or not adapter._state_valid:
        raise ValueError("Diagnostic checkpoint changed or is invalid")
    control = adapter._checked_controller
    control.world.refresh()
    before = state_signature(adapter, control)
    start = control.world.positions()
    scene, _ = attached_scene(control.world, start, control.contact_grants)
    scene = control._contact_scene(scene)
    initial_check = control._public_check({"detail": scene.check(
        start, allowed_pairs=control.allowed_pairs, clearance_limit=.001)})
    report = {"status": "running", "model_api_calls": 0, "physics_advanced": False,
              "scope": plan["scope"], "state_before": before, "initial_check": initial_check,
              "probes": []}
    path = Path(output) / "endpoint-probes.json"
    write_json(path, report)
    deadline = time.monotonic() + plan["total_timeout_s"]
    had_action_override = "_action" in control.__dict__
    original_override = control.__dict__.get("_action")

    def no_action(*args, **kwargs):
        raise RuntimeError("Physical execution is forbidden during endpoint diagnosis")

    control._action = no_action
    try:
        for probe in plan["probes"]:
            if time.monotonic() >= deadline:
                report["status"] = "diagnostic_timeout"
                break
            role_index = control.roles.index(probe["role"])
            poses = [control.tcp(i).copy() for i in range(len(control.roles))]
            poses[role_index] = np.asarray(probe["target"], dtype=float)
            started = time.monotonic()
            goals, detail = combined_goals(
                control, poses, start, scene, [role_index],
                min(deadline, started + plan["per_probe_timeout_s"]),
            )
            after = state_signature(adapter, control)
            unchanged = after == before
            report["probes"].append({
                **probe, "endpoint_admissible": bool(goals),
                "receipt": control._public_plan(detail),
                "elapsed_s": time.monotonic() - started, "state_unchanged": unchanged,
                "physics_steps": after["physics_step_counter"] - before["physics_step_counter"],
            })
            report["state_after"] = after
            report["physics_advanced"] = after["physics_step_counter"] != before["physics_step_counter"]
            write_json(path, report)
            if not unchanged:
                raise RuntimeError("State changed during endpoint-only checks")
        else:
            report["status"] = "completed"
    except BaseException:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        raise
    finally:
        if had_action_override:
            control._action = original_override
        else:
            del control._action
        report["state_after"] = state_signature(adapter, control)
        report["state_unchanged"] = report["state_after"] == before
        report["physics_advanced"] = report["state_after"]["physics_step_counter"] != before["physics_step_counter"]
        write_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker"))
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--after-call", type=int)
    parser.add_argument("--packet", type=Path)
    args, worker_args = parser.parse_known_args()
    sys.path.insert(0, str(args.reference_root.resolve()))
    if args.mode == "prepare":
        from tools.codex_robodojo_host import target_matrix
        plan = make_probes(args.baseline, args.after_call,
                           json.loads(args.packet.read_text()), target_matrix)
        validate_plan(plan)
        write_json(args.output / "probe-request.json", plan)
        return
    from sim.envs.robodojo import worker_bridge
    original_operate = worker_bridge.operate
    consumed = False

    def operate(adapter, body):
        nonlocal consumed
        result = original_operate(adapter, body)
        plan_path = args.output / "probe-request.json"
        if body.get("operation") == "status" and plan_path.exists() and not consumed:
            consumed = True
            run_probes(adapter, json.loads(plan_path.read_text()), args.output)
        return result

    worker_bridge.operate = operate
    sys.argv = [str(args.reference_root / "sim/bench_worker.py"), *worker_args]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
