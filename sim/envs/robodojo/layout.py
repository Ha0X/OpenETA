"""Import-safe native layout contracts; support actors are never target effectors."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

COMPETITION = 'dual_x5_and_franka_competition'
SUPPORT_POLICY = 'native_auxiliary_ownership_v1'
SUPPORT_REQUIREMENTS = ['support_articulation_geometry', 'time_dependent_collision_check',
                        'native_support_phase_advancement', 'observable_demonstration_progress']


def task_robot_name(task, settings):
    common = settings.get('common') or {}
    override = (settings.get('tasks') or {}).get(task) or {}
    if not isinstance(common, dict) or not isinstance(override, dict):
        raise ValueError('Malformed native task robot declaration')
    name = override.get('robot_config', common.get('robot_config'))
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('Missing or invalid native robot configuration; support-robot requirements unknown')
    return name


def describe_layout(document):
    robots = document.get('robots')
    if not isinstance(robots, list) or not robots or not all(isinstance(r, dict) for r in robots):
        raise ValueError('Robot configuration requires a nonempty robot list')
    names = tuple(r.get('robot_name') for r in robots)
    types = tuple(r.get('type', 'target') for r in robots)
    if names == ('x5',) and types == ('target',):
        name, roles, support = 'single_x5', ['primary'], {}
    elif names == ('x5', 'x5') and types == ('target', 'target'):
        name, roles, support = 'dual_x5', ['left', 'right'], {}
    elif names == ('x5', 'x5', 'franka') and types == ('target', 'target', 'support'):
        name, roles = COMPETITION, ['left', 'right']
        support = {'support_arm0': {'model':'franka', 'control_mode':'task_managed',
                                    'agent_controllable':False}}
    else:
        raise ValueError('Unsupported robot order, model or target/support classification')
    return {'layout':name, 'target_roles':roles, 'support_actors':support,
            'robot_config_sha256':hashlib.sha256(json.dumps(document, sort_keys=True,
                                                          separators=(',', ':')).encode()).hexdigest()}


def resolve_native_layout(root, task, settings, load_yaml):
    name = task_robot_name(task, settings)
    document = load_yaml(str(Path(root) / 'env_cfg' / 'robot' / (name + '.yml')))
    return {'robot_config':name, **describe_layout(document)}


def bind_layout(native, effective_document, variant, *, support_policy=None):
    effective = describe_layout(effective_document)
    contract = {'schema':'openeta.robodojo_layout.v1', 'native':native, 'effective':effective,
                'execution_variant':variant, 'experimental_override':False,
                'missing_capabilities':[]}
    if support_policy not in (None, SUPPORT_POLICY):
        raise ValueError('Unknown auxiliary ownership policy')
    if native['support_actors'] and support_policy == SUPPORT_POLICY:
        if (native['layout'] != COMPETITION or effective['layout'] != COMPETITION
                or variant != 'dual_x5'
                or native['robot_config_sha256'] != effective['robot_config_sha256']):
            raise ValueError('Auxiliary execution requires the unchanged native competition layout')
        contract['support_policy'] = SUPPORT_POLICY
        return contract
    if native['support_actors']:
        contract['missing_capabilities'] = SUPPORT_REQUIREMENTS.copy()
        raise ValueError('support-robot execution is not admitted: '+json.dumps(contract, sort_keys=True))
    if native['layout'] != 'dual_x5' or variant not in ('single_x5', 'dual_x5'):
        raise ValueError('Unsupported native/task execution binding')
    if effective['layout'] != variant:
        raise ValueError('Loaded robot layout differs from requested execution variant')
    # Preserve the existing explicit single-arm experiment; never present it as native dual.
    contract['experimental_override'] = variant == 'single_x5'
    return contract


def verify_live_layout(env, contract):
    robots = getattr(getattr(env, 'robot_manager', None), 'robot_list', None)
    roles = contract['effective']['target_roles']
    support = contract['effective']['support_actors']
    if support and (contract.get('support_policy') != SUPPORT_POLICY
                    or contract['effective']['layout'] != COMPETITION):
        raise RuntimeError('Live auxiliary layout lacks an explicit ownership contract')
    if robots is None or len(robots) != len(roles)+len(support):
        raise RuntimeError('Live robot count differs from the bound layout')
    expected = ['target']*len(roles)+['support']*len(support)
    if [getattr(robot, 'type', 'target') for robot in robots] != expected:
        raise RuntimeError('Unexpected target/support control ownership in live execution')
    env._openeta_layout_contract = contract
