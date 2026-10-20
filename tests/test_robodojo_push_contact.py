from types import SimpleNamespace
import numpy as np
import pytest
from test_robodojo_tool_contact import stroke_setup
from sim.envs.robodojo.push_contact import PushContact, execute_push
from sim.envs.robodojo.support_contact import SupportContactScene


def setup():
    c,t=stroke_setup()
    from sim.envs.robodojo.scene_geometry import forward_kinematics
    from sim.envs.robodojo.x5_calibration import link_to_tcp
    c.tcp=lambda _:link_to_tcp(forward_kinematics(c.world.scene.snapshot['robots'][0],c.world.positions()[0])['link6'])
    t={'left':[c.tcp(0)]};t['left'][0][0,3]+=.04
    c.contact_grants={}
    return c,t


def make(c,t,**kw):
    return PushContact(c,'left',t,body_point=[-1,0,0],surface_point=[-1,0,-.04],
                       target_points=[[-1,.09,-.02]],**kw)


def test_no_grasp_needed_and_pair_scope_is_explicit():
    c,t=setup();p=make(c,t,fingers=['finger_1'])
    assert ('finger7','held') in p.pairs and ('finger8','held') not in p.pairs
    assert ('finger7','workpiece') not in p.pairs and ('finger7','table') not in p.pairs
    assert ('held','workpiece') in p.pairs
    assert p.public()['rigid_attachment_assumed'] is False
    assert c.contact_grants=={}
    assert np.array_equal(p.effective,t['left'][0])


def test_selected_bodies_can_slide_relative_to_wrist_without_attachment_failure():
    c,t=setup();p=make(c,t)
    transform=np.eye(4);transform[0,3]=.025
    c.world.scene.world_transforms[2]=transform;c.world.scene.set_transform(2,transform)
    assert p.check(c.world.scene,c.world.positions(),measured=True)['status']=='clear'
    assert p.public()['object_effect'].startswith('unverified')


def test_penetration_and_patch_still_stop_actual_motion():
    c,t=setup();p=make(c,t)
    transform=np.eye(4);transform[2,3]=-.012
    c.world.scene.world_transforms[2]=transform;c.world.scene.set_transform(2,transform)
    assert p.check(c.world.scene,c.world.positions(),measured=True)['reason']=='push_support_penetration_limit'
    transform=np.eye(4);transform[0,3]=.5
    c.world.scene.world_transforms[2]=transform;c.world.scene.set_transform(2,transform)
    assert p.check(c.world.scene,c.world.positions(),measured=True)['reason']=='push_body_left_authorized_patch'


def test_no_robot_workpiece_collision_exemption():
    c,t=setup();p=make(c,t)
    transform=np.eye(4);transform[1,3]=-.04
    c.world.scene.world_transforms[4]=transform;c.world.scene.set_transform(4,transform)
    assert not SupportContactScene(c.world.scene,p).check(c.world.positions())['clear']


def test_bounds_and_other_arm_ownership_are_not_relaxed():
    c,t=setup();t['left'][0][0,3]+=.02
    with pytest.raises(ValueError,match='5cm'):make(c,t)
    c,t=setup();c.contact_grants={'right':{'root':'body','indices':[2]}}
    with pytest.raises(ValueError,match='another hand'):make(c,t)
    c,t=setup()
    with pytest.raises(ValueError,match='Select finger'):make(c,t,fingers=[])
    t['left'].append(t['left'][0])
    with pytest.raises(ValueError,match='one short target'):make(c,t)


def test_explicit_push_revokes_old_grasp_and_cleans_up_even_after_stop(monkeypatch):
    c,t=setup();c.contact_grants={'left':{'root':'body','indices':[2],'pairs':[]}}
    c.cloth=None;c.support_holds={};c.allowed_pairs=[]
    c._expire_support=lambda roles,reason:None;c._rebuild_allowed_pairs=lambda:None
    from sim.envs.robodojo import motion_planning
    def execute(control,targets,**kw):
        from sim.envs.robodojo.attached_geometry import attached_scene
        scene,evidence=attached_scene(c.world,c.world.positions(),c.contact_grants)
        assert scene is c.world.scene and not evidence
        assert control.support_placement is not None
        return {'planning':{},'host_adjustments':[],'stop':{'reason':'test_stop'}}
    monkeypatch.setattr(motion_planning,'execute_request',execute)
    result=execute_push(c,t,{'body_point':[-1,0,0],'surface_point':[-1,0,-.04],
                       'target_points':[[-1,.09,-.02]]},timeout_s=10,max_actions=10)
    assert result['host_adjustments'][0]['previous_grasp_authorization_revoked']
    assert c.support_placement is None and not c.contact_grants
