from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.checked_control import CheckedController, cartesian_knots, pose_error


def controller(roles):
    c=CheckedController.__new__(CheckedController)
    c.roles=roles
    c.adapter=SimpleNamespace(_terminal=False)
    c.native=SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[object() for _ in roles]))
    schema={'joint_names':[f'j{i}' for i in range(8)],'arm_joint_names':[f'j{i}' for i in range(6)]}
    c.allowed_pairs=[]
    c.contact_grants={}
    c.world=SimpleNamespace(positions=lambda:[np.full(8,.2+i) for i in range(len(roles))],refresh=lambda:None,
                           scene=SimpleNamespace(snapshot={'robots':[schema for _ in roles]},check=lambda *a,**kw:{'clear':True,'clearance_m':1.}),public_policy=lambda:{})
    c.poses=[np.eye(4) for _ in roles]
    c.tcp=lambda i:c.poses[i].copy()
    c._grippers=lambda:[1. for _ in roles]
    return c


def test_unreachable_destination_rejected_before_any_physics_or_partial_approach():
    c=controller(['primary'])
    c._solve_pose=lambda *a:(None,{'reason':'ik_unreachable','role':'primary'})
    def fail(*a):raise AssertionError('An unreachable destination must not dispatch')
    c._action=fail
    target=np.eye(4);target[0,3]=.1
    result=c._recovery_move({'primary':[target]})
    assert result['native_actions']==result['physics_steps']==0
    assert result['stop']['phase']=='endpoint_preview'
    assert not result['reached_target']


def test_unselected_arm_keeps_measured_joint_target_without_new_ik_branch():
    c=controller(['left','right']);solves=[];actions=[]
    target=np.eye(4);target[0,3]=.01
    def solve(i,pose,current):
        solves.append(i)
        return np.full(6,.3),None
    def action(q,g):
        actions.append(q);c.poses[0]=target.copy();c.physics_steps+=1
    c._solve_pose=solve;c._action=action
    result=c._recovery_move({'left':[target]})
    assert result['reached_target']
    assert solves==[0,0]
    assert np.allclose(actions[0][1],np.full(6,1.2))
    assert result['unselected_effectors']=='held_at_initial_measured_joint_targets'


def test_cartesian_knots_bound_position_and_orientation_changes():
    start=np.eye(4);target=np.diag([-1.,-1.,1.,1.]);target[0,3]=.15
    knots=cartesian_knots(start,target)
    for a,b in zip([start,*knots[:-1]],knots):
        error=pose_error(a,b)
        assert error['position_m']<=.02+1e-8
        assert error['orientation_rad']<=.10+1e-8
    assert np.allclose(knots[-1],target)


def test_support_robot_cannot_be_addressed_through_target_command_or_ik():
    import pytest
    c = CheckedController.__new__(CheckedController)
    c.roles = ['left', 'right']
    with pytest.raises(ValueError, match='target effectors'):
        c._action([np.zeros(6)] * 3, [1.] * 3)
    with pytest.raises(ValueError, match='target robots'):
        c._pose_candidates(2, np.eye(4), np.zeros(7))
    with pytest.raises(ValueError, match='target robots'):
        c.tcp(2)
