"""Read owned X5 effector joints through the native measured-state API.

No commands, object poses, inferred grasps, or simulator steps are produced.
The reported opening uses measured physical joint limits when available. The
native command range may extend past a physical stop; it must not make an
empty, fully closed gripper appear partly open. This is not an aperture in m.
"""
from __future__ import annotations

import numpy as np


def _physical_limits(manager, robot, joint_name):
    entities = getattr(manager, 'robot_key', None)
    if entities is None:
        return None
    index = next(i for i, candidate in enumerate(manager.robot_list) if candidate is robot)
    entity = entities[index]
    names = list(entity.joint_names)
    if names.count(joint_name) != 1:
        raise ValueError('physical_joint_mapping_ambiguous')
    limits = entity.data.joint_pos_limits
    if hasattr(limits, 'detach'):
        limits = limits.detach().cpu().numpy()
    limits = np.asarray(limits, dtype=float)
    if limits.shape != (1, len(names), 2):
        raise ValueError('physical_limits_require_one_owned_environment')
    scale = limits[0, names.index(joint_name)]
    if not np.isfinite(scale).all() or scale[1] <= scale[0]:
        raise ValueError('invalid_physical_joint_limits')
    return scale


def measure_effectors(env, roles):
    result = {role: {'measurement_available': False, 'measurement_reason': 'native_measurement_api_unavailable'}
              for role in roles}
    manager = getattr(env, 'robot_manager', None)
    read = getattr(manager, 'get_end_effector_real_val', None)
    if not callable(read):
        return result
    robots = [robot for robot in getattr(manager, 'robot_list', []) if getattr(robot, 'type', None) == 'target']
    if len(robots) != len(roles):
        return {role: {'measurement_available':False,'measurement_reason':'native_target_robot_count_mismatch'} for role in roles}
    mapped = {}
    for robot in robots:
        role = 'primary' if len(roles) == 1 else str(getattr(robot,'arm_name','')).removesuffix('_arm')
        if (role not in roles or role in mapped or getattr(robot,'robot_name',None) != 'x5'
                or getattr(robot,'ee_type',None) != 'gripper'):
            return {role: {'measurement_available':False,'measurement_reason':'native_effector_identity_mismatch'} for role in roles}
        mapped[role] = robot
    for role,robot in mapped.items():
        try:
            measured = read(robot, env_idx_list=[0])
            # Do not select whichever environment happens to occur first.
            if not isinstance(measured, dict) or 0 not in measured:
                raise ValueError('owned_environment_measurement_missing')
            values = np.asarray(measured[0], dtype=float)
            names = getattr(robot, 'gripper_joints_name', None)
            if (not isinstance(names, (list,tuple)) or not names or any(not isinstance(n,str) or not n for n in names)
                    or len(set(names)) != len(names) or values.shape != (len(names),) or not np.isfinite(values).all()):
                raise ValueError('invalid_measured_joint_mapping')
            item = {'measurement_available': True, 'measurement_source':'native_joint_pos',
                    'joint_positions':dict(zip(names,values.tolist())), 'environment_index':0}
            scale = np.asarray(getattr(robot,'gripper_scale',None), dtype=float)
            sign = getattr(robot,'gripper_move',{}).get('sign')
            if (scale.shape != (2,) or not np.isfinite(scale).all() or scale[1] <= scale[0]
                    or type(sign) not in (int,float) or sign not in (-1,1)):
                item['normalization_reason'] = 'gripper_model_limits_or_sign_unavailable'
            else:
                model_scale = scale.copy()
                try:
                    physical_scale = _physical_limits(manager, robot, names[0])
                except (AttributeError, ValueError, IndexError, TypeError, StopIteration) as exc:
                    item['normalization_reason'] = f'physical_limits_unverified:{type(exc).__name__}'
                    result[role] = item
                    continue
                basis = 'first_actuated_joint_normalized'
                if physical_scale is not None:
                    scale = physical_scale
                    basis = 'first_actuated_joint_physical_limits'
                    item['normalization'] = {
                        'joint_name': names[0], 'physical_joint_limits': scale.tolist(),
                        'command_joint_limits': model_scale.tolist(), 'sign': sign,
                    }
                fraction = float((values[0]-scale[0])/(scale[1]-scale[0]))
                openness = fraction if sign == 1 else 1-fraction
                if 0 <= openness <= 1:
                    item.update(openness=openness, openness_basis=basis)
                else:
                    item.update(normalization_reason='measured_joint_outside_nominal_range',
                                unclipped_normalized_joint=openness)
            result[role] = item
        except (AttributeError, TypeError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            # Failure of this optional sensor path must not fabricate closed
            # fingers or erase the command-preservation state.
            result[role] = {'measurement_available':False,
                            'measurement_reason':f'native_measurement_failed:{type(exc).__name__}'}
    return result
