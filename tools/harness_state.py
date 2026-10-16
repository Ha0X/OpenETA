"""Named embodiment observations and shared-joint scope analysis.

Used by new adapters before exposing calibrated task tools. The legacy Panda
interface keeps its original codec. Unknown calibration is reported, never
replaced with Panda geometry or an identity transform.
"""
from __future__ import annotations

import math

import numpy as np

from tools.harness_config import ConfigError


def vector(value, count, where):
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ConfigError('invalid_observed_state', where) from exc
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ConfigError('invalid_observed_state', f'{where} must be a finite {count}-vector')
    return result.tolist()


def reported_pose(value):
    if isinstance(value, dict):
        xyz = vector(value.get('xyz'), 3, 'position')
        quat = vector(value.get('quat_xyzw'), 4, 'quaternion')
    else:
        pose = vector(value, 7, 'pose')
        xyz, quat = pose[:3], pose[3:]
    norm = math.sqrt(sum(v*v for v in quat))
    if norm < 1e-8:
        raise ConfigError('invalid_observed_state', 'Zero quaternion is not an orientation')
    return {'xyz': xyz, 'quat_xyzw': [v/norm for v in quat]}


def named_robot_state(config, observation):
    body = config.value['components']['embodiment']
    root = observation.get('robot', observation.get('proprio'))
    if not isinstance(root, dict):
        raise ConfigError('missing_robot_state', 'Observation has no robot/proprio dictionary')
    roles = body['effectors']
    if 'arms' in root:
        entries = root['arms']
        if not isinstance(entries, dict) or set(entries) != set(roles):
            raise ConfigError('observed_effector_mismatch', 'Observed arm roles differ from the bound embodiment')
    elif len(roles) == 1:
        entries = {next(iter(roles)): root}
    else:
        raise ConfigError('missing_named_effectors', 'A flat joint vector cannot identify multiple arms')
    result = {}
    for role, definition in roles.items():
        entry = entries[role]
        if not isinstance(entry, dict):
            raise ConfigError('invalid_observed_state', role)
        pose = reported_pose(entry.get('end_effector_pose', entry.get('ee_pose')))
        joints = entry.get('joint_positions')
        if joints is None:
            joints = []
        if not isinstance(joints, (list, tuple, np.ndarray)):
            raise ConfigError('invalid_observed_state', f'{role} joint positions')
        joints = vector(joints, len(joints), f'{role} joint positions')
        group = body['groups'][definition['group']]
        names = entry.get('joint_names')
        expected = group['joint_names']
        if names is not None:
            if (not isinstance(names, list) or len(names) != len(joints) or
                    any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names)):
                raise ConfigError('invalid_observed_state', f'{role} joint name mapping')
            if expected is not None and set(names) != set(expected):
                raise ConfigError('observed_joint_mismatch', role)
        # Never invent names for a deployment whose joint ordering is unknown.
        named_joints = dict(zip(names, joints)) if names is not None else None
        actuator = {'kind': definition['kind']}
        if definition['kind'] == 'parallel_gripper':
            grip = entry.get('gripper_state', {})
            measured = isinstance(grip, dict) and grip.get('measurement_available') is True
            if isinstance(grip, dict):
                for key in ('measurement_available', 'measurement_source', 'measurement_reason',
                            'normalization_reason', 'openness_basis', 'position_error_from_command'):
                    if key in grip:
                        actuator[key] = grip[key]
                actual_joints = grip.get('joint_positions')
                if measured and isinstance(actual_joints, dict) and actual_joints:
                    values = vector(list(actual_joints.values()), len(actual_joints), 'effector measured joints')
                    actuator['joint_positions'] = dict(zip(actual_joints, values))
            command = grip.get('commanded_openness') if isinstance(grip, dict) else None
            if command is not None:
                command = float(command)
                if not math.isfinite(command) or not 0 <= command <= 1:
                    raise ConfigError('invalid_observed_state', 'Commanded gripper opening must be in [0,1]')
                actuator['last_commanded_openness'] = command
            openness = grip.get('openness') if isinstance(grip, dict) else None
            if openness is None:
                openness = entry.get('gripper_open')
            if openness is not None:
                if isinstance(openness, (bool, np.bool_)):
                    raise ConfigError('invalid_observed_state', 'Boolean openness cannot substitute for measured aperture')
                openness = float(openness)
                if not math.isfinite(openness) or not 0 <= openness <= 1:
                    raise ConfigError('invalid_observed_state', 'Gripper openness must be measured in [0,1]')
                actuator['openness' if measured else 'reported_openness_unverified'] = openness
                actuator['measurement_available'] = measured
            else:
                actuator['state'] = 'measured_joints_only' if measured else 'unknown'
        elif definition['kind'] == 'dexterous_hand':
            # Preserve actual finger positions; no scalar gripper conversion.
            fingers = entry.get('finger_joint_positions')
            if isinstance(fingers, dict) and fingers:
                names_hand = list(fingers)
                values = vector(list(fingers.values()), len(fingers), 'finger positions')
                actuator['joint_positions'] = dict(zip(names_hand, values))
            else:
                actuator['state'] = 'unknown'
        else:
            actuator['state'] = 'unsupported_effector_codec'
        result[role] = {'group': definition['group'], 'reported_pose': pose,
            'reported_position_frame': definition['reported_position_frame'],
            'reported_orientation_frame': definition['reported_orientation_frame'],
            'tcp_calibration_available': definition['calibration'] is not None,
            'joints_by_name': named_joints, 'joint_positions_reported': joints,
            'effector': actuator}
    return {'schema_version': 'openeta.named_robot_state.v1', 'effectors': result}


def resolve_motion_scope(config, *, target_effectors, participating_groups):
    """Resolve resources without expanding requested scope or solving motion.

Shared joints identify additional TCPs whose hold constraints require a joint
solver. The caller must validate that solver's capability before execution.
"""
    parts = config.value['components']
    body = parts['embodiment']
    for names in (target_effectors, participating_groups):
        if (not isinstance(names, (list, tuple)) or not names
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names)):
            raise ConfigError('invalid_motion_scope', 'Use nonempty unique lists of named resources')
    targets, groups = set(target_effectors), set(participating_groups)
    if not targets or targets - set(body['effectors']):
        raise ConfigError('unknown_target_effector', 'Select named effectors from the bound config')
    if not groups or groups - set(parts['execution']['scope']):
        raise ConfigError('motion_scope_not_authorized', 'Requested groups exceed the bound execution scope')
    required = {body['effectors'][role]['group'] for role in targets}
    if required - groups:
        raise ConfigError('target_group_not_participating', 'Selected effectors require their own control groups')
    if any(value['joint_names'] is None for value in body['groups'].values()):
        raise ConfigError('joint_resource_mapping_unknown', 'Cannot resolve shared resources without joint names')
    group_joints = {name: [(value.get('articulation'),joint) for joint in value['joint_names']]
                    for name,value in body['groups'].items()}
    label = lambda joint: joint[1] if joint[0] is None else f'{joint[0]}::{joint[1]}'
    moving = set().union(*(set(group_joints[group]) for group in groups))
    affected = {role for role, definition in body['effectors'].items()
                if groups & set(definition['affected_by_groups'])
                or moving & set(group_joints[definition['group']])}
    holds = affected - targets
    shared = {joint: sorted(name for name, values in group_joints.items() if joint in values)
              for joint in moving}
    shared = {label(joint): names for joint,names in shared.items() if len(names) > 1}
    return {'target_effectors': sorted(targets), 'participating_groups': sorted(groups),
            'joint_write_set': sorted(label(j) for j in moving), 'shared_joints': shared,
            'hold_effectors': sorted(holds), 'requires_coupled_solver': bool(shared or holds),
            'requires_atomic_execution': len(groups) > 1,
            'scope_expanded': False, 'execution_authorized': False}
