from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.checked_control import CheckedController
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_planning_feedback import data


def controller(granted):
    source = data()
    source['colliders'][1]['vertices'] = (np.array(source['colliders'][1]['vertices'])+[-11.,0,0]).tolist()
    scene = CollisionScene(source)
    c = CheckedController.__new__(CheckedController)
    c.roles = ['left']; c.allowed_pairs = []
    c.native = SimpleNamespace(sim=SimpleNamespace(physics_dt=.004))
    c.gripper_intents = [0.]
    c.contact_grants = {'left': {'root': 'body', 'indices': [1], 'pairs': []}} if granted else {}
    c.world = SimpleNamespace(scene=scene, positions=lambda: [np.array([-1.])], refresh=lambda: None,
                             support_pairs=[], public_policy=lambda: {}, table_id='other_table')
    c.adapter = SimpleNamespace(_terminal=False, _last_obs={'proprio': {'arms': {
        'left': {'gripper_state': {'openness': 0.}}}}})
    return c


def test_opening_authorized_wrist_object_contact_can_enter_bounded_release():
    c = controller(True)
    def action(targets, grippers):
        assert c.recovery.release
        assert c.recovery.pairs
        c.adapter._last_obs['proprio']['arms']['left']['gripper_state']['openness'] = 1.
        c.physics_steps += 1
    c._action = action
    result = c.gripper('left', 1., max_actions=1)
    assert result['recovery_mode'] == 'bounded_release'
    assert result['native_actions'] == 1
    assert not c.contact_grants


def test_opening_does_not_authorize_an_unrelated_wrist_collision():
    c = controller(False)
    def forbidden(*args):
        raise AssertionError('Unrelated contact may not receive a release exception')
    c._action = forbidden
    result = c.gripper('left', 1.)
    assert result['native_actions'] == result['physics_steps'] == 0
    assert result['stop']['reason'] == 'gripper_initial_collision'
    assert c.gripper_intents == [0.]
    assert result['gripper_intent_latched'] is False


def test_closing_does_not_receive_the_opening_recovery_exception():
    c = controller(True)
    c._action = lambda *args: (_ for _ in ()).throw(AssertionError('Closing must not dispatch'))
    result = c.gripper('left', 0.)
    assert result['native_actions'] == 0
    assert result['stop']['reason'] == 'gripper_initial_collision'
