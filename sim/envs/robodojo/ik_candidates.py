"""cuRobo v2 candidate generation; physical USD FK/collision owns admission.

The native planner's synthetic table and incomplete robot spheres must not
decide reachability. This separate solver changes no native planner or physics
state. A failed finite search is unknown, not proof of unreachability.
"""
from copy import deepcopy

import numpy as np

from .scene_geometry import array
from .x5_calibration import tcp_to_link


class CuroboCandidates:
    def __init__(self, native, snapshot):
        self.native = native
        self.snapshot = snapshot
        self.solvers = {}

    def solve(self, index, tcp, current):
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import GoalToolPose, JointState, Pose
        from scipy.spatial.transform import Rotation

        schema = self.snapshot['robots'][index]
        if index not in self.solvers:
            robot = self.native.robot_manager.robot_list[index]
            source = self.native.robot_manager.ik_solver[robot.robot_name]
            cfg = InverseKinematicsCfg.create(
                robot=deepcopy(source.robot_cfg), scene_model=None,
                device_cfg=source.device_cfg, num_seeds=32, seed_solver_num_seeds=32,
                position_tolerance=.001, orientation_tolerance=.02,
                self_collision_check=False, use_cuda_graph=True,
                max_batch_size=1, multi_env=False, max_goalset=1)
            self.solvers[index] = InverseKinematics(cfg)
        solver = self.solvers[index]
        link = tcp_to_link(tcp).copy()
        link[:3, 3] += array(self.native.robot_manager.scene.env_origins)[0]
        local = np.linalg.inv(array(schema['body_frames'][schema['base_link']])) @ link
        xyzw = Rotation.from_matrix(local[:3, :3]).as_quat()
        goal = GoalToolPose.from_poses({solver.tool_frames[0]: Pose.from_numpy(
            local[:3, 3], np.r_[xyzw[3], xyzw[:3]])}, num_goalset=1)
        names = list(solver.joint_names)
        values = array([current[schema['joint_names'].index(n)] for n in names])
        import torch
        state = JointState.from_position(torch.as_tensor(values, device='cuda', dtype=torch.float32)[None],
                                         joint_names=names)
        result = solver.solve_pose(goal_tool_poses=goal, current_state=state, return_seeds=32)
        success = array(result.success).astype(bool).reshape(-1)
        candidates = ordered_seed_candidates(result.js_solution, success, schema['arm_joint_names'])
        detail = {'backend': 'curobo_v2_kinematic_candidates',
            'attempted_seeds': 32, 'converged_seeds': int(success.sum()),
            'returned_seed_candidates': len(candidates),
            'solver_success_flag_is_admission': False,
            'kinematic_admission': 'physical_joint_limits_and_usd_fk_target_tolerances',
            'collision_admission': 'physical_usd_geometry',
            'search_exhaustion_is_unreachability_proof': False}
        if not success.any():
            detail['sampled_residuals'] = sampled_seed_residuals(
                schema, current, candidates, tcp,
                array(self.native.robot_manager.scene.env_origins)[0])
        return candidates, detail



def ordered_seed_candidates(js, success, arm_joint_names):
    """Map every solver proposal; success flags are diagnostics, not admission.

    A solver failure may still have an exact physical solution. Conversely a
    solver success can violate actual asset limits/FK. Both require the same
    downstream physical validation and collision checks before any execution.
    """
    names = list(js.joint_names or [])
    if not names or len(names) != len(set(names)):
        raise ValueError('cuRobo candidate identity contract changed')
    positions = array(js.position)
    if positions.shape[-1:] != (len(names),):
        raise ValueError('cuRobo candidate joint shape contract changed')
    positions = positions.reshape(-1, len(names))
    if len(success) != len(positions):
        raise ValueError('cuRobo candidate seed shape contract changed')
    indices = [names.index(n) for n in arm_joint_names]
    return positions[:, indices]


def validated_candidates(schema, current, candidates, target, origin):
    """Name mapping happens before this boundary; verify limits and physical FK."""
    from scipy.spatial.transform import Rotation
    from .scene_geometry import forward_kinematics
    from .x5_calibration import link_to_tcp
    indices = [schema['joint_names'].index(n) for n in schema['arm_joint_names']]
    limits = array(schema['joint_limits'])[indices]
    valid, errors = [], {'nonfinite_or_shape': 0, 'joint_limit': 0, 'physical_fk': 0}
    for raw in candidates:
        q = array(raw).copy()
        if q.shape != (len(indices),) or not np.isfinite(q).all():
            errors['nonfinite_or_shape'] += 1
            continue
        for j, k in enumerate(indices):
            joint = next(item for item in schema['joints'] if item['name'] == schema['joint_names'][k])
            if joint['kind'] != 'revolute':
                continue
            turn = int(round((current[k]-q[j])/(2*np.pi)))
            choices = [q[j]+2*np.pi*n for n in {0, turn-1, turn, turn+1}]
            choices = [v for v in choices if limits[j, 0]-1e-7 <= v <= limits[j, 1]+1e-7]
            if choices:
                q[j] = min(choices, key=lambda v: abs(v-current[k]))
        if np.any(q < limits[:, 0]-1e-7) or np.any(q > limits[:, 1]+1e-7):
            errors['joint_limit'] += 1
            continue
        full = array(current).copy()
        full[indices] = q
        actual = link_to_tcp(forward_kinematics(schema, full)['link6'])
        actual[:3, 3] -= origin
        pe = np.linalg.norm(actual[:3, 3]-target[:3, 3])
        re = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
        if pe > .003 or re > .03:
            errors['physical_fk'] += 1
            continue
        if not any(np.max(abs(q-other)) < 1e-4 for other in valid):
            valid.append(q)
    valid.sort(key=lambda q: float(np.linalg.norm(q-array(current)[indices])))
    return valid, errors


def sampled_seed_residuals(schema, current, candidates, target, origin):
    """Explain finite failed IK samples; never admit them or expose alternative goals.

    Each reported error pair belongs to one physically joint-limit-valid sample.
    These samples have NOT passed collision or target-tolerance admission.
    """
    from scipy.spatial.transform import Rotation
    from .scene_geometry import forward_kinematics
    from .x5_calibration import link_to_tcp

    indices = [schema['joint_names'].index(n) for n in schema['arm_joint_names']]
    residuals = []
    for raw in candidates:
        q = array(raw)
        if q.shape != (len(indices),) or not np.isfinite(q).all():
            continue
        full = array(current).copy()
        full[indices] = q
        self_pose = link_to_tcp(forward_kinematics(schema, full)['link6'])
        self_pose[:3, 3] -= origin
        # Reuse exactly the same periodic normalization and physical limits as
        # admission. Validating against the sample's own FK admits no motion.
        valid, _ = validated_candidates(schema, current, [q], self_pose, origin)
        if not valid:
            continue
        full[indices] = valid[0]
        actual = link_to_tcp(forward_kinematics(schema, full)['link6'])
        actual[:3, 3] -= origin
        residuals.append({
            'position_error_m': float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            'orientation_error_rad': float(Rotation.from_matrix(
                target[:3, :3] @ actual[:3, :3].T).magnitude())})
    result = {'physical_limit_valid_samples': len(residuals),
        'collision_checked': False, 'target_changed': False,
        'limitation': 'Finite sampled residuals do not prove unreachability. Each error pair belongs to one sample; none is an executable alternative pose.'}
    if residuals:
        result['best_balanced_sample'] = min(residuals, key=lambda r:
            (r['position_error_m']/.003)**2 + (r['orientation_error_rad']/.03)**2)
        result['smallest_position_error_sample'] = min(residuals, key=lambda r: r['position_error_m'])
        result['smallest_orientation_error_sample'] = min(residuals, key=lambda r: r['orientation_error_rad'])
    return result
