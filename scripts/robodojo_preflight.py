#!/usr/bin/env python3
"""No-side-effect RoboDojo installation and manifest preflight."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def check(output: Path, robot_variant: str = "single_x5") -> int:
    repo = Path(__file__).resolve().parents[1]
    manifest_path = repo / "sim" / "envs" / "robodojo" / ("dual_task_manifest.json" if robot_variant == "dual_x5" else "task_manifest.json")
    result: dict = {"backend": "robodojo", "checks": [], "ok": True}

    def add(name: str, ok: bool, detail: str = ""):
        result["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        result["ok"] &= bool(ok)

    add("python", sys.version_info >= (3, 11), sys.version.split()[0])
    add("nvidia_smi", shutil.which("nvidia-smi") is not None,
        shutil.which("nvidia-smi") or "missing")
    root = os.environ.get("ROBODOJO_ROOT", "")
    add("ROBODOJO_ROOT", bool(root and Path(root).exists()), root or "unset")
    asset_root = os.environ.get("ROBODOJO_ASSET_ROOT", "")
    add("ROBODOJO_ASSET_ROOT", bool(asset_root and Path(asset_root).exists()), asset_root or "unset")
    robot_config = os.environ.get(
        "ROBODOJO_ROBOT_CONFIG",
        str(repo / "sim" / "envs" / "robodojo" / "config" / ("dual_x5.yml" if robot_variant == "dual_x5" else "single_x5.yml")),
    )
    add(
        f"{robot_variant}_robot_config",
        bool(robot_config and Path(robot_config).exists())
        if os.path.isabs(robot_config)
        else bool(robot_config and root and (Path(root) / "env_cfg" / "robot" / f"{robot_config}.yml").exists()),
        robot_config,
    )
    try:
        import yaml
        robot_doc = yaml.safe_load(Path(robot_config).read_text(encoding="utf-8"))
        robots = robot_doc.get("robots", []) if isinstance(robot_doc, dict) else []
        expected = 2 if robot_variant == "dual_x5" else 1
        add(f"{robot_variant}_robot_count", len(robots) == expected, f"count={len(robots)} expected={expected}")
    except Exception as exc:
        add(f"{robot_variant}_robot_count", False, str(exc))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        add("manifest", manifest.get("schema_version") == "1", str(manifest_path))
        expected = 2 if robot_variant == "dual_x5" else 1
        add(f"{robot_variant}_tasks", all(int(t.get("required_arms", expected)) == expected for t in manifest.get("tasks", [])),
            str(len(manifest.get("tasks", []))))
    except Exception as exc:
        add("manifest", False, str(exc))
    add("isaac_import_in_host", "isaacsim" not in sys.modules,
        "host process must not import Isaac Sim")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/robodojo/preflight.json"))
    parser.add_argument("--robot-variant", choices=("single_x5", "dual_x5"), default="single_x5")
    args = parser.parse_args()
    raise SystemExit(check(args.output, args.robot_variant))
