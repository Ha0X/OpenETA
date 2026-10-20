from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.separating_recovery import certify_upward


def scene(kind='prismatic'):
    robot={'base_link':'base','joint_names':['lift'],'joint_limits':[[-1.,1.]],
        'body_frames':{'base':np.eye(4).tolist(),'finger':np.eye(4).tolist()},
        'joints':[{'name':'lift','parent':'base','child':'finger','kind':kind,'axis':'Z',
                   'parent_frame':np.eye(4).tolist(),'child_frame':np.eye(4).tolist()}]}
    return SimpleNamespace(snapshot={'robots':[robot]},items=[{'robot_index':0,'body':'finger'}],
        vertices=[np.array([[.01,0,0],[0,.01,.01]])],bounds=[1.])


def test_upward_separation_is_certified_but_downward_and_tangential_are_not():
    assert certify_upward(scene(),[[0]],[[.01]],{0})['status']=='clear'
    assert certify_upward(scene(),[[0]],[[-.01]],{0})['status']=='blocked'
    assert certify_upward(scene('revolute'),[[0]],[[.1]],{0})['status']=='blocked'


def test_recovery_budget_failure_stays_unknown():
    assert certify_upward(scene(),[[0]],[[.01]],{0},max_nodes=0)['status']=='unknown'


def test_endpoint_lift_is_not_enough_when_joint_arc_dips_midway():
    s=scene('revolute')
    s.snapshot['robots'][0]['joints'][0]['axis']='Y'
    s.vertices=[np.array([[.01,0,0],[.02,0,0]])]
    # z=-r*sin(q): q=4 is above q=0, but the arc first passes below it.
    result=certify_upward(s,[[0]],[[4.]],{0})
    assert result['status']=='blocked'
    assert result['reason']=='recovery_would_deepen_finger_table_contact'
