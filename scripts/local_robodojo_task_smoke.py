"""One real scene reset, camera capture and bounded controller check; no model."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.multi_gpu = False
    args.output.mkdir(parents=True, exist_ok=False)
    from sim import bench_worker as worker
    report = {"task": args.task, "seed": 0, "kind": "environment_smoke_not_model_attempt",
              "ok": False, "phase": "startup", "job_id": os.getenv("SLURM_JOB_ID")}
    env = None

    def plain(value):
        if hasattr(value, "detach"):
            return value.detach().cpu().tolist()
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value

    def save():
        (args.output / "result.json").write_text(json.dumps(plain(report), indent=2) + "\n")
        print("TASK_SMOKE", report["phase"], flush=True)

    try:
        save()
        worker._init_robodojo_runtime(args)
        report["phase"] = "create"; save()
        env = worker._make_env(f"openeta/robodojo_dual_x5_{args.task}-v0", seed=0)
        report["phase"] = "reset"; save()
        observation, info = env.reset(seed=0)
        report["instruction"] = observation["task_description"]
        report["reset_info"] = info
        (args.output / "task.txt").write_text(report["instruction"] + "\n")
        import numpy as np
        from PIL import Image
        cameras = {}
        for i, (name, camera) in enumerate(observation["cameras"].items()):
            rgb = np.asarray(camera["rgb"])
            assert rgb.ndim == 3 and rgb.shape[-1] >= 3 and np.std(rgb) > 1, (name, rgb.shape)
            Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(args.output / f"camera-{i}.png")
            cameras[name] = {"shape": list(rgb.shape), "image": f"camera-{i}.png"}
        assert cameras, "No camera observations"
        report["cameras"] = cameras
        report["phase"] = "controller"; save()
        from sim.envs.robodojo.checked_control import CheckedController
        control = CheckedController(env._env, allow_articulations=True, allow_fluid=True)
        report["roles"] = control.roles
        before = control.tcp(1).copy()
        target = control.tcp(0).copy()
        target[2, 3] += 0.02
        report["phase"] = "left_arm_lift_2cm"; save()
        report["motion"] = control.move({"left": [target]})
        report["right_arm_pose_delta"] = float(np.max(np.abs(control.tcp(1) - before)))
        assert report["motion"]["reached_target"], report["motion"]
        assert report["right_arm_pose_delta"] < 0.002
        report.update(ok=True, phase="complete")
    except BaseException:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    finally:
        save()
        if env is not None:
            env.close()
        process = worker._robodojo_policy_process
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill(); process.wait(timeout=10)
        if worker._robodojo_app is not None:
            worker._robodojo_app.close()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
