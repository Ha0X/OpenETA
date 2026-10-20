"""Opt-in semantic oracle. Native predicate inputs never cross this module's boundary.

Snapshot only; no reward-manager step/check/get_reward and no simulator mutation.
This profile is intentionally limited to the reviewed native build_tower revision.
"""
import copy
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROFILE = 'build_tower_semantic_v1'
PINS = {
    'task': '814cf6ffb459871e5cf49f4e127903dbfbc30926f33db71de0b84a71ab235d6e',
    'parser': 'eb928a05c3eb6bae429a922c59e64272c2c3851c22d779143d45627e47b3a504',
    'manager': 'bc56efd3aaa861b4b38960dc4fb7778a9b4da9cf131f815ac3170300dd233a41',
}
ALLOWED = {'is_axis_up', 'is_axis_aligned', 'is_A_up_B',
           'is_A_in_B_support_circle', 'all_robot_back_to_origin'}


def _verify(env):
    objects = {'task':env._tower_completion_checks, 'parser':type(env.reward_manager.func_parser),
               'manager':type(env.reward_manager)}
    for key, obj in objects.items():
        if hashlib.sha256(Path(inspect.getsourcefile(obj)).read_bytes()).hexdigest() != PINS[key]:
            raise ValueError('Unsupported native predicate revision')


def _finite(value, shape):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Invalid snapshot')
    return a.copy()


class FrozenLayout:
    def __init__(self, native):
        self.poses = {}; self.supports = {}
        for label in [f'block{i}' for i in range(8)]:
            name = native.get_instance_name(label=label, env_idx=0)
            if name is None or native.get_scene_object(inst_name=name, env_idx=0) is None:
                raise ValueError('Missing object')
            pos, quat = native.get_instance_pose(inst_name=name, env_idx=0)
            pos, quat = _finite(pos, (3,)), _finite(quat, (4,))
            if abs(np.linalg.norm(quat)-1) > .01:
                raise ValueError('Invalid quaternion')
            self.poses[label] = (pos, quat)
            if label in ('block0','block3','block7'):
                points, radii = native.get_support_points(tag='block/0', type='passive',
                    config=native.get_instance_metadata(inst_name=name, env_idx=0),
                    ret='list', obj_name=name, env_idx=0)
                if not points or radii is None or len(points)!=len(radii):
                    raise ValueError('Missing support metadata')
                if not all(np.isfinite(np.asarray(p,dtype=float)).all() for p in points):
                    raise ValueError('Invalid support metadata')
                self.supports[label] = copy.deepcopy((points, radii))
    def get_instance_name(self, *, label, env_idx):
        assert env_idx == 0
        return label if label in self.poses else None
    def get_instance_pose(self, *, inst_name, env_idx):
        assert env_idx == 0
        return copy.deepcopy(self.poses[inst_name])
    def get_scene_object(self, *, inst_name, env_idx):
        assert env_idx == 0
        return inst_name if inst_name in self.poses else None
    def get_instance_metadata(self, *, inst_name, env_idx):
        assert env_idx == 0
        return {}
    def get_support_points(self, *, obj_name, tag, type, ret, config, env_idx):
        assert env_idx == 0 and tag == 'block/0' and type == 'passive' and ret == 'list'
        return copy.deepcopy(self.supports[obj_name])


class FrozenRobots:
    def __init__(self, parser):
        self.robot_list=[]; self.poses={}; self.origins=[{}]
        for robot in parser.robot_manager.robot_list:
            if robot.type != 'target':
                continue
            name=robot.arm_name
            self.robot_list.append(SimpleNamespace(type='target',arm_name=name))
            self.poses[name]=_finite(parser.robot_manager.get_real_endpose(robot)[0],(7,))
            self.origins[0][name]=_finite(parser.robot_origin_endpose[0][name],(7,))
        if len(self.robot_list)!=2:
            raise ValueError('Expected two target arms')
    def get_real_endpose(self, robot):
        return [self.poses[robot.arm_name].copy()]


def _snapshot(parser):
    robots=FrozenRobots(parser)
    # No live env, simulator or native manager is reachable by the receiver.
    return SimpleNamespace(num_envs=1,layout_manager=FrozenLayout(parser.layout_manager),
        robot_manager=robots,robot_origin_endpose=robots.origins,
        _select_label=lambda args: args['label'])


def _groups(env):
    groups=[]
    for i, noun in enumerate(('long board','support block','support block','thin top plank',
                              'green top piece','support block','support block','short board')):
        groups.append(('object_orientation',f'The {noun} does not have the required upright orientation.',
                       [env._axis_up_check(f'block{i}')]))
    groups.extend([
        ('base_structure','The long board and lower supports do not satisfy the required layer relationship.',env._base_structure_checks()),
        ('middle_structure','The upper supports and boards do not satisfy the required layer or centering relationship.',env._middle_structure_checks()),
        ('thin_plank_height','The thin top plank is not above the short board as required.',[env._a_up_b('block3','block7',z_threshold_min=.012)]),
        ('top_piece_height','The green top piece is not above the thin plank as required.',[env._a_up_b('block4','block3',z_threshold_min=.015)]),
        ('thin_plank_centering','The thin top plank is off-center relative to the short board.',[env._support_circle_check('block3','block7',radius=.023)]),
        ('top_piece_centering','The green top piece is off-center relative to the thin plank.',[env._support_circle_check('block4','block3',radius=.023)]),
        ('top_alignment','The thin plank, short board and green top piece do not have the required relative orientation.',env._tower_alignment_checks()),
        ('robot_return','One or more arms have not returned to their initial position and orientation.',[env.reward_manager.all_robot_back_to_origin()]),
    ])
    # Assert the diagnostic covers exactly the official completion tree, with
    # the same leaf arguments. No independently invented success criterion.
    expected=[*env._tower_completion_checks(),env.reward_manager.all_robot_back_to_origin()]
    def leaves(node):
        if isinstance(node,tuple):return [repr(node)]
        return [x for child in node for x in leaves(child)]
    if sorted(leaves(expected)) != sorted(leaves([g[2] for g in groups])):
        raise ValueError('Diagnostic/native condition coverage differs')
    return groups, expected


def _node(node, parser_type, frozen, op='or'):
    if isinstance(node, tuple):
        name,args=node
        if name not in ALLOWED:
            raise ValueError('Unreviewed predicate')
        value=getattr(parser_type,name)(frozen,dict(copy.deepcopy(args),env_idx=0))
        if value not in (0.,1.):raise ValueError('Invalid predicate result')
        return bool(value)
    if not isinstance(node,list) or not node:
        raise ValueError('Invalid predicate tree')
    values=[_node(child,parser_type,frozen,'and' if op=='or' else 'or') for child in node]
    return any(values) if op=='or' else all(values)


def evaluate_groups(groups, expected, parser_type, frozen):
    def conjunction(tree):return all(_node(n,parser_type,frozen) for n in tree)
    failures=[]
    for code,message,tree in groups:
        if not conjunction(tree) and (code,message) not in failures:failures.append((code,message))
    satisfied=conjunction(expected)
    if satisfied != (not failures):raise ValueError('Diagnostic grouping differs')
    return satisfied,[{'code':c,'message':m} for c,m in failures]


def check(adapter, requested_index):
    if requested_index != adapter._step_count:
        raise ValueError('Stale completion check; observe again')
    result={'schema':'openeta.semantic_completion_check.v1','profile':PROFILE,
        'feedback_source':'privileged_simulator_semantic_oracle',
        'source_action_index':adapter._step_count,'native_actions':0,'physics_steps':0,
        'official_success_recorded':bool(adapter._env.end_flag[0] and adapter._env.success[0]),
        'validity':'Snapshot only; reassess after any physical action. This check does not advance physics or award success.'}
    try:
        if adapter._task_name!='build_tower' or adapter._robot_variant!='dual_x5' or not adapter._reward_registered:
            raise ValueError('Unsupported task')
        env=adapter._env;_verify(env)
        groups,expected=_groups(env)
        frozen=_snapshot(env.reward_manager.func_parser)
        satisfied,issues=evaluate_groups(groups,expected,type(env.reward_manager.func_parser),frozen)
        result.update(status='checked',current_conditions_satisfied=satisfied,issues=issues)
    except Exception:
        # Never serialize native exceptions, labels, poses, thresholds or trees.
        result.update(status='unavailable',current_conditions_satisfied=None,issues=[],
                      reason='The read-only checker could not verify this native state or revision.')
    return result
