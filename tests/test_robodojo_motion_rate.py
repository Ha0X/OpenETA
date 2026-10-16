from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.checked_control import CheckedController
from sim.envs.robodojo.motion_rate import joint_step_limits


def control():
    c = CheckedController.__new__(CheckedController)
    c.roles = ['left', 'right']
    c.operation = 'move'
    c.recovery = None
    c.contact_grants = {}
    return c


@pytest.mark.parametrize('grippers,grants,expected', [
    ([1.,1.], {}, [.08,.08]), ([1.,0.], {}, [.08,.04]),
    ([.98,1.], {}, [.04,.08]), ([1.,1.], {'left':{}}, [.04,.08]),
    ([0.,0.], {}, [.04,.04]),
])
def test_no_attachment_evidence_does_not_speed_up_closed_or_authorized_hands(grippers, grants, expected):
    c=control();c.contact_grants=grants
    assert joint_step_limits(c,grippers)==expected


@pytest.mark.parametrize('attribute,value,expected', [
    ('operation','gripper',.04), ('recovery',object(),.02),
    ('support_placement',SimpleNamespace(),.01),
    ('support_placement',SimpleNamespace(workpieces={}),.01),
])
def test_contact_and_recovery_caps_override_open_command(attribute,value,expected):
    c=control();setattr(c,attribute,value)
    assert joint_step_limits(c,[1.,1.])==[expected,expected]


@pytest.mark.parametrize('grippers', [[1.], [1.,float('nan')], [1.,1.01]])
def test_invalid_commands_reject_before_dispatch(grippers):
    with pytest.raises(ValueError):joint_step_limits(control(),grippers)


@pytest.mark.parametrize('contact', [False, True])
def test_mixed_hand_limits_preserve_common_path_fraction_and_physics_guard(contact):
    c=control();schema={'joint_names':['a','b'],'arm_joint_names':['a','b']}
    if contact:
        c.support_placement = SimpleNamespace(workpieces={}, role='left')
    c.world=SimpleNamespace(positions=lambda:[np.zeros(2),np.zeros(2)],
        scene=SimpleNamespace(snapshot={'robots':[schema,schema]}))
    c.joint_step_limited_actions=0
    calls=[];guard=[]
    c.native=SimpleNamespace(sim_step=lambda: calls.append('physics'))
    original=c.native.sim_step
    c._guarded_step=lambda fn,*a,**k:(guard.append('checked'),fn(*a,**k))
    c.gripper_continuity=SimpleNamespace(action=lambda _:nullcontext())
    actions=[]
    def step(action):
        actions.append(action);c.native.sim_step()
    c.adapter=SimpleNamespace(step=step)
    c._action([np.array([.2,.1]),np.array([.2,.1])],[1.,0.])
    # The holding right arm limits both arms to a common 0.2 path fraction.
    assert np.allclose(c.arm_targets,[[.01,.005],[.01,.005]] if contact else [[.04,.02],[.04,.02]])
    assert calls==['physics'] and guard==['checked']
    assert c.native.sim_step is original
    if contact:
        assert c.motion_rate_events == {}
    else:
        assert c.motion_rate_events['left']['condition']=='fully_open_command_and_no_grasp_contact_authorization'
    assert 'right' not in c.motion_rate_events
    c._action([np.array([.2,.1]),np.zeros(2)],[1.,0.])
    assert np.allclose(c.arm_targets,[[.01,.005],[0.,0.]] if contact else [[.08,.04],[0.,0.]])
    assert actions[-1]['arms']['right']['gripper']==0.
