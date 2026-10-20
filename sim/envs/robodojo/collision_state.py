"""Collision-only prismatic finger state from live physical link poses.

PhysX can report a finger DOF at its limit while its actual link extends past
that limit. Reconstruct the axial coordinate, never move bodies or alter drives.
Full FK validation still runs with the original tolerance afterwards.
"""
import numpy as np
from .scene_geometry import array, pose_matrix


def measured_finger_positions(schema, positions, measured_frames):
    result = array(positions).copy()
    for name in schema.get('finger_links', ()):
        joints = [j for j in schema['joints'] if j['child'] == name]
        if (len(joints) != 1 or joints[0]['kind'] != 'prismatic'
                or joints[0]['name'] in schema.get('arm_joint_names', ())):
            raise ValueError('Collision finger requires a verified non-arm prismatic physical joint')
        joint = joints[0]
        motion = (np.linalg.inv(array(joint['parent_frame']))
                  @ np.linalg.inv(measured_frames[joint['parent']])
                  @ measured_frames[name] @ array(joint['child_frame']))
        axis = np.eye(3)['XYZ'.index(joint['axis'])]
        displacement = float(motion[:3,3] @ axis)
        residual = motion[:3,3]-axis*displacement
        if (not np.isfinite(motion).all() or np.max(abs(residual)) > 2e-5
                or not np.allclose(motion[:3,:3], np.eye(3), atol=2e-4, rtol=0.)):
            raise ValueError('Collision finger pose violates its physical prismatic joint')
        result[schema['joint_names'].index(joint['name'])] = displacement
    return result


def auxiliary_finger_positions(schema, positions, measured_frames):
    """Compatibility helper for callers explicitly inspecting auxiliary state."""
    if schema.get('control_owner') != 'support':
        return array(positions).copy()
    return measured_finger_positions(schema, positions, measured_frames)


def stationary_measured_finger_slots(robot, start, target):
    """Unchanged measured geometry is not a commanded joint-limit violation.

    Internal collision paths start from live verified coordinates. Only a
    carried-through prismatic finger qualifies; changed targets and arm joints
    retain their limits. Actual geometry and swept bounds remain unclipped.
    """
    fingers = set(robot.get('finger_links', ()))
    arms = set(robot.get('arm_joint_names', ()))
    result = set()
    for joint in robot['joints']:
        if (joint['kind'] == 'prismatic' and joint['child'] in fingers
                and joint['name'] not in arms):
            slot = robot['joint_names'].index(joint['name'])
            if start[slot] == target[slot]:
                result.add(slot)
    return result


def collision_positions(robots, entities):
    result=[]
    for schema, entity in zip(robots, entities):
        q=array(entity.data.joint_pos)[0].copy()
        poses=array(entity.data.body_link_pose_w)[0]
        frames={name:pose_matrix(poses[i]) for i,name in enumerate(entity.body_names)}
        q=measured_finger_positions(schema,q,frames)
        result.append(q)
    return result
