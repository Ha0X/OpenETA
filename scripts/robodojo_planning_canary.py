"""Operator canary for an isolated deployed RoboDojo controller release.

Run with the deployed RoboDojo Python and environment, --headless
--enable_cameras. Output is operator-private; it is not an agent observation.
"""
import argparse
import json
import sys
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--task', default='general_pickup')
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    sys.path.insert(0, args.release)
    from sim import bench_worker as worker
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    report = {'purpose': 'physical_dual_planning_canary', 'task': args.task, 'passed': False}
    env = None
    def plain(value):
        if hasattr(value, 'detach'):
            return value.detach().cpu().tolist()
        if hasattr(value, 'tolist'):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        return value
    def save():
        (output/'report.json').write_text(json.dumps(plain(report), indent=2)+'\n')
    try:
        report['phase'] = 'initializing'; save()
        worker._init_robodojo_runtime(args)
        env = worker._make_env('openeta/robodojo_dual_x5_'+args.task+'-v0', seed=0)
        env.reset(seed=0)
        from sim.envs.robodojo.checked_control import CheckedController
        control = CheckedController(env._env)
        report['roles'] = control.roles
        report['sphere_validation'] = control.world.sphere_validation
        report['phase'] = 'ik_candidates'; save()
        q = control.world.positions()
        report['candidate_checks'] = []
        for i in range(2):
            solutions, info = control._pose_candidates(i, control.tcp(i), q[i])
            report['candidate_checks'].append(info); save()
            assert solutions, info
        report['phase'] = 'left_only'; save()
        right_before = control.tcp(1)
        left = control.tcp(0); left[2, 3] += .02
        report['left_only'] = control.move({'left': [left]}); save()
        assert report['left_only']['reached_target'], report['left_only']
        import numpy as np
        report['unselected_right_matrix_delta'] = float(np.max(abs(control.tcp(1)-right_before)))
        assert report['unselected_right_matrix_delta'] < .002
        report['phase'] = 'both'; save()
        left = control.tcp(0); left[2, 3] += .02
        right = control.tcp(1); right[2, 3] += .03
        report['both'] = control.move({'left': [left], 'right': [right]}); save()
        assert report['both']['reached_target'], report['both']
        report['left_close'] = control.gripper('left', 0.); save()
        report['left_open'] = control.gripper('left', 1.); save()
        assert report['left_close']['measured']['openness'] < .02
        assert report['left_open']['measured']['openness'] > .98
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
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
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
