"""Operator-only evidence for a failed FK check; never repairs/admission.

Use the exact frames already compared by NativeScene, without another physics
step or state read. Local joint residuals distinguish an inherited world-frame
error from a discrepancy at that joint. They do not establish a physical cause.
"""
import numpy as np

from .scene_geometry import array, forward_kinematics


def _safe(value):
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def _residual(expected, measured):
    finite = bool(np.isfinite(expected).all() and np.isfinite(measured).all())
    return {
        'finite': finite,
        # Keep the exact existing admission predicate, including NumPy rtol.
        'matches_existing_guard': bool(np.allclose(expected, measured, atol=2e-4)),
        'translation_error_m': float(np.linalg.norm(expected[:3, 3]-measured[:3, 3])),
        'rotation_matrix_max_error': float(abs(expected[:3, :3]-measured[:3, :3]).max()),
    }


def fk_mismatch_report(robot, positions, measured_frames):
    """JSON-safe, robot-only report. No objects, contacts or reward state."""
    q = array(positions).copy()
    predicted = forward_kinematics(robot, q)
    measured = {name: array(frame).copy() for name, frame in measured_frames.items()}
    if set(predicted) != set(measured):
        raise ValueError('FK diagnostic body coverage differs')
    base = robot['base_link']
    links = []
    for name, frame in predicted.items():
        residual = _residual(frame, measured[name])
        if not residual['matches_existing_guard']:
            links.append({'link': name, **residual,
                          'predicted_frame': frame, 'measured_frame': measured[name]})
    joints = []
    for joint in robot['joints']:
        parent, child = joint['parent'], joint['child']
        p, c = array(joint['parent_frame']), array(joint['child_frame'])
        if not all(np.isfinite(x).all() for x in (p, c, measured[parent], measured[child])):
            joints.append({'joint': joint['name'], 'kind': joint['kind'], 'finite': False})
            continue
        expected = np.linalg.inv(p) @ np.linalg.inv(predicted[parent]) @ predicted[child] @ c
        actual = np.linalg.inv(p) @ np.linalg.inv(measured[parent]) @ measured[child] @ c
        item = {'joint': joint['name'], 'kind': joint['kind'], 'parent': parent, 'child': child,
                **_residual(expected, actual)}
        if joint['kind'] == 'prismatic':
            axis = np.eye(3)['XYZ'.index(joint['axis'])]
            displacement = float(actual[:3, 3] @ axis)
            position = float(q[robot['joint_names'].index(joint['name'])])
            item.update(collision_coordinate_m=position, measured_axial_coordinate_m=displacement,
                        axial_coordinate_error_m=displacement-position,
                        transverse_error_m=float(np.linalg.norm(actual[:3, 3]-axis*displacement)))
        joints.append(item)
    return _safe({
        'schema_version': 'openeta.fk_diagnostic.v1',
        'scope': 'operator_only_robot_kinematics',
        'role': robot.get('role'), 'control_owner': robot.get('control_owner'),
        'collision_positions': dict(zip(robot['joint_names'], q.tolist())),
        'base': {'link': base, **_residual(predicted[base], measured[base])},
        'mismatched_links': links, 'local_joint_residuals': joints,
        'interpretation': 'Residuals describe this sample, not a confirmed physical cause. '
                          'No tolerance, drive, joint limit or collision geometry was changed.',
    })
