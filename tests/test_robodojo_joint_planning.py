from copy import deepcopy

import numpy as np
import pytest

from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.joint_planning import JointPlanner
from sim.envs.robodojo.boundary_recovery import BoundaryRecovery
from test_robodojo_collision_geometry import fixture


def test_joint_planner_rejects_obstacle_even_when_endpoint_clear():
    scene = CollisionScene(fixture())
    path, receipt = JointPlanner(scene, [0]).plan([[-1.]], [[[1.]]], max_nodes=10)
    assert path is None and receipt['status'] == 'unknown'
    assert receipt['search_dof'] == 1


def test_joint_planner_certifies_every_returned_segment():
    scene = CollisionScene(fixture())
    path, receipt = JointPlanner(scene, [0]).plan([[-1.]], [[[-.3]]])
    assert receipt['status'] == 'clear'
    assert all(scene.check_path(a, b)['status'] == 'clear' for a, b in zip(path, path[1:]))


def dual_crossing():
    data = fixture(start=-.3, wall_center=10.)
    second = deepcopy(data['robots'][0])
    second['joints'][0]['axis'] = 'Y'
    second['body_frames']['tool'][0][3] = 0.
    second['body_frames']['tool'][1][3] = -.3
    data['robots'].append(second)
    tip = deepcopy(data['colliders'][0]); tip.update(id='second_finger', robot_index=1)
    data['colliders'].append(tip)
    return CollisionScene(data)


def test_two_individually_clear_moves_collide_when_executed_together():
    scene = dual_crossing()
    start = [[-.3], [-.3]]
    assert scene.check_path(start, [[.3], [-.3]])['status'] == 'clear'
    assert scene.check_path(start, [[-.3], [.3]])['status'] == 'clear'
    assert scene.check_path(start, [[.3], [.3]])['status'] != 'clear'
    path, detail = JointPlanner(scene, [0, 1]).plan(start, [[[.3], [.3]]], max_nodes=256, timeout_s=5.)
    assert path is not None and detail['search_dof'] == 2
    assert all(scene.check_path(a, b)['status'] == 'clear' for a, b in zip(path, path[1:]))


def test_unselected_arm_cannot_change_in_planner_goal():
    with pytest.raises(ValueError, match='held joint'):
        JointPlanner(dual_crossing(), [0]).plan([[-.3], [-.3]], [[[.3], [.1]]])


def test_recovery_distinguishes_separating_deepening_and_plateau():
    scene = CollisionScene(fixture(start=-.08))
    contacts = scene.check([[-.08]])['contacts']
    assert contacts
    recovery = BoundaryRecovery(scene, [[-.08]], contacts)
    assert recovery.check_segment([[-.08]], [[-.12]])['status'] == 'clear'
    assert recovery.check_segment([[-.08]], [[-.04]])['status'] == 'blocked'
    assert recovery.check_segment([[-.08]], [[-.08]])['status'] == 'blocked'
    release = BoundaryRecovery(scene, [[-.08]], contacts, release=True)
    assert release.check_segment([[-.08]], [[-.08]])['status'] == 'clear'
    assert not release.check_actual([[-.04]])
    assert release.last_check['maximum_deepening_m'] > .01
    assert release.last_check['contacts'][0]['measured_signed_clearance_m'] < release.last_check['contacts'][0]['previous_signed_clearance_m']


def test_budget_never_claims_motion_clear():
    path, detail = JointPlanner(dual_crossing(), [0, 1]).plan([[-.3], [-.3]], [[[.3], [.3]]], timeout_s=0.)
    assert path is None and detail['status'] == 'unknown'


@pytest.mark.parametrize('total_limit',[.25,15.])
def test_longer_edge_budget_still_obeys_total_deadline(monkeypatch,total_limit):
    scene=CollisionScene(fixture());original=scene.check_path;limits=[]
    def check(*args,**kwargs):
        limits.append(kwargs['timeout_s'])
        return original(*args,**kwargs)
    monkeypatch.setattr(scene,'check_path',check)
    path,receipt=JointPlanner(scene,[0],edge_timeout_s=4.).plan([[-1.]],[[[-.7]]],timeout_s=total_limit)
    assert path is not None and limits
    assert all(0<limit<=min(4.,total_limit) for limit in limits)
    assert receipt['edge_check_time_limit_s']==4.


def test_shared_body_constraint_rejects_stretching_even_without_geometry_collision():
    data = deepcopy(dual_crossing().snapshot)
    for robot in data['robots']:
        robot['role'] = 'left' if robot is data['robots'][0] else 'right'
        robot['body_frames']['link6'] = robot['body_frames'].pop('tool')
        robot['joints'][0]['child'] = 'link6'
    for item in data['colliders']:
        if item['body'] == 'tool':
            item['body'] = 'link6'
    a, b = [np.array(r['body_frames']['link6']) for r in data['robots']]
    data['closed_chain_constraints'] = [{'owner': 0, 'follower': 1,
                                        'relative_pose': (np.linalg.inv(a) @ b).tolist()}]
    scene = CollisionScene(data)
    assert scene.check([[-.3],[-.3]])['clear']
    result = scene.check_path([[-.3],[-.3]], [[-.4],[-.3]])
    assert result['reason'] == 'shared_body_relative_wrist_constraint'
