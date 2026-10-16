from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.checked_control import CheckedController
from sim.envs.robodojo.boundary_recovery import BoundaryRecovery
from test_robodojo_attached_geometry import grasp_scene


def test_recovery_refreshes_attachment_evidence_and_geometry():
    world, grants = grasp_scene()
    c = CheckedController.__new__(CheckedController)
    c.world = world
    c.contact_grants = grants
    c.operation = 'move'
    c.recovery = SimpleNamespace(scene=object())
    first, evidence = c._prediction_scene([[-1., 0., 0.]])
    assert evidence and c.recovery.scene is first
    assert first.items[2]['robot_index'] == 0
    # Fingers separate from the authorised body: the next recovery step must
    # not keep the old attached shape or suppress loss-of-evidence monitoring.
    second, evidence = c._prediction_scene([[-1., .03, 0.]])
    assert not evidence and second is world.scene
    assert c.recovery.scene is second
    assert second.items[2]['robot_index'] is None


def test_release_does_not_reintroduce_rigid_attachment_to_opening_hand():
    world, grants = grasp_scene()
    c = CheckedController.__new__(CheckedController)
    c.world = world; c.contact_grants = grants; c.operation = 'gripper'
    c.recovery = SimpleNamespace(scene=world.scene)
    scene, evidence = c._prediction_scene([[-1., 0., 0.]])
    assert scene is world.scene and not evidence


def test_contact_trace_exposes_best_so_far_reference_and_stays_bounded():
    values = iter([-.00025, -.000243, -.000265])
    scene = SimpleNamespace(items=[{'id':'a','robot_index':None}, {'id':'table','robot_index':None}],
        signed_distances=lambda *args:{frozenset(('a','table')):next(values)})
    recovery = BoundaryRecovery(scene, [], [{'a':'a','b':'table'}])
    assert recovery.check_actual([])
    assert not recovery.check_actual([])
    report = recovery.diagnostics()
    assert report['measured_steps'] == 2
    assert abs(report['recent_samples'][-1]['maximum_deepening_from_best_m']-.000022)<1e-10
    scene.signed_distances=lambda *args:{frozenset(('a','table')):-.000243}
    for _ in range(100):recovery.check_actual([])
    assert len(recovery.diagnostics()['recent_samples']) == 64
