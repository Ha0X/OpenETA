from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest

from sim.envs.robodojo.physical_contact import PhysicalContact
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_collision_geometry import fixture


def world(body='link7'):
    data=fixture(start=-.3)
    robot=data['robots'][0]
    robot['body_frames'][body]=robot['body_frames'].pop('tool')
    robot['joints'][0]['child']=body
    data['colliders'][0]['body']=body
    return SimpleNamespace(scene=CollisionScene(data),support_pairs=[])


def test_finger_contact_needs_no_mark_or_intent_but_has_measured_bound():
    w=world();p=PhysicalContact(w, roles=['left']);p.begin([[-.3]])
    assert not w.scene.check([[-.0995]])['clear']
    assert w.scene.check([[-.0995]],allowed_pairs=p.pairs)['clear']
    assert p.check([[-.0995]])['clear']
    result=p.check([[-.098]])
    assert not result['clear'] and result['reason']=='physical_contact_penetration_limit'
    assert 'finger' not in json.dumps(result.get('detail',{}))
    assert p.public()['force_control'] is False
    assert p.public()['held_object_prediction'] is False


def test_non_finger_contact_stays_protected():
    w=world('link6');p=PhysicalContact(w, roles=['left'])
    assert p.pairs==[]
    assert not w.scene.check([[-.0995]],allowed_pairs=p.pairs)['clear']


def test_inter_arm_finger_contact_is_never_exempted():
    w=world();data=deepcopy(w.scene.snapshot)
    data['robots'].append(deepcopy(data['robots'][0]))
    tip=deepcopy(data['colliders'][0]);tip.update(id='other_finger',robot_index=1)
    data['colliders'].append(tip)
    w.scene=CollisionScene(data);p=PhysicalContact(w, roles=['left','right'])
    assert not w.scene.check([[-.3],[-.3]],allowed_pairs=p.pairs)['clear']
    assert frozenset(['finger','other_finger']) not in p.keys


def test_recovery_is_nonworsening_and_cannot_reuse_old_deeper_bound():
    p=PhysicalContact(world(), roles=['left']);p.begin([[-.098]])
    assert p.recovery and p.check([[-.098]])['clear']
    assert p.check([[-.0995]])['clear']
    assert not p.check([[-.098]])['clear']
    p.begin([[-.098]])
    assert not p.check([[-.096]])['clear']


def test_unavailable_contact_distance_does_not_claim_clear(monkeypatch):
    w=world();p=PhysicalContact(w, roles=['left']);p.begin([[-.3]])
    def fail(*a,**kw):raise ValueError('private geometry detail')
    monkeypatch.setattr(w.scene,'signed_distances',fail)
    result=p.check([[-.0995]])
    assert not result['clear'] and 'private' not in json.dumps(result)


@pytest.mark.parametrize('fail_on',[1,2])
def test_contact_guard_checks_before_and_after_actual_physics(monkeypatch,fail_on):
    import sim.envs.robodojo.attached_geometry as attachments
    from test_robodojo_gripper_feedback import controller
    c=controller();c.operation='move';c.stopped=None;c.physics_steps=0;c.recovery=None
    c.attachment_baseline={};c.attachment_monitor={'roles_with_evidence':[],'checked_physics_steps':0,'relative_motion':{}}
    c.arm_start=[np.array([0.])];c.arm_targets=[np.array([0.])]
    c.world.scene.snapshot['robots'][0]['joint_limits']=[[-1.,1.]]
    c.world.scene.check_path=lambda *a,**k:{'status':'clear'}
    c.native.robot_manager=SimpleNamespace(robot_key=[SimpleNamespace(data=SimpleNamespace(joint_pos_target=np.array([[0.]])))])
    c._contact_scene=lambda s:s;c._active_contacts=lambda:[];c._workpiece_collision_pairs=lambda:[]
    c._prediction_scene=lambda current:(c.world.scene,[])
    monkeypatch.setattr(attachments,'attachment_frames',lambda *args:{})
    calls=[];physical=[]
    def check(q):
        calls.append(1)
        return {'clear':len(calls)!=fail_on,'reason':'physical_contact_penetration_limit'}
    c.physical_contact=SimpleNamespace(check=check)
    c._halt=lambda reason,detail=None:setattr(c,'stopped',{'reason':reason})
    c._guarded_step(lambda:physical.append(1))
    assert len(physical)==c.physics_steps==fail_on-1
    assert c.stopped['reason']=='physical_contact_penetration_limit'
    c._guarded_step(lambda:physical.append(1))
    assert len(physical)==fail_on-1


def test_bridge_cannot_silently_enable_contact_policy_after_startup():
    from test_robodojo_workpiece_bridge import setup_bridge
    from sim.envs.robodojo.worker_bridge import operate
    a,b,_,_=setup_bridge();b['contact_policy']='physical_contact_v1'
    with pytest.raises(ValueError,match='initial status'):operate(a,b)
    a._checked_contact_policy='physical_contact_v1';b.pop('contact_policy')
    with pytest.raises(ValueError,match='within an episode'):operate(a,b)


def test_contact_bound_names_actual_fingers_without_private_geometry_or_depth():
    w=world();data=deepcopy(w.scene.snapshot)
    other=deepcopy(data['robots'][0]);other['body_frames']['link8']=other['body_frames'].pop('link7');other['joints'][0]['child']='link8'
    data['robots'].append(other)
    tip=deepcopy(data['colliders'][0]);tip.update(id='private_other_finger',robot_index=1,body='link8')
    data['colliders'].append(tip);w.scene=CollisionScene(data)
    p=PhysicalContact(w,roles=['left','right']);p.begin([[-.3],[-.3]])
    result=p.check([[-.3],[-.098]])
    pairs=result['contact_bound_violations']
    assert [x['participants'][0] for x in pairs]==[{'kind':'robot','role':'right','part':'finger_2'}]
    assert pairs[0]['participants'][1]=={'kind':'scene_geometry'}
    assert 'private_other_finger' not in json.dumps(result) and 'depth' not in json.dumps(result)
    p.begin([[-.3],[-.3]]);result=p.check([[-.098],[-.098]])
    assert [x['participants'][0]['role'] for x in result['contact_bound_violations']]==['left','right']


def test_contact_roles_must_match_robot_identity():
    for roles in ([],['left','right'],['']):
        with pytest.raises(ValueError,match='public role'):PhysicalContact(world(),roles=roles)


def test_agent_owned_franka_fingers_use_bounded_contact_without_exempting_hand():
    w=world('panda_leftfinger')
    w.scene.snapshot['robots'][0]['finger_links']=['panda_leftfinger','panda_rightfinger']
    p=PhysicalContact(w,roles=['support_arm0']);p.begin([[-.3]])
    assert p.pairs and p.check([[-.0995]])['clear']
    result=p.check([[-.098]])
    assert not result['clear']
    assert result['contact_bound_violations'][0]['participants'][0]=={'kind':'robot','role':'support_arm0','part':'finger_1'}
    w=world('panda_hand');w.scene.snapshot['robots'][0]['finger_links']=['panda_leftfinger','panda_rightfinger']
    assert PhysicalContact(w,roles=['support_arm0']).pairs==[]


def test_native_auxiliary_contacts_are_delegated_without_agent_contact_permissions():
    w=world();data=deepcopy(w.scene.snapshot);r=data['robots'][0]
    r.update(control_owner='support',finger_links=['link7','link8'])
    r['body_frames']['link8']=deepcopy(r['body_frames']['link7'])
    joint=deepcopy(r['joints'][0]);joint.update(name='fixed_second',parent='link7',child='link8',kind='fixed',collision_enabled=True)
    r['joints'].append(joint)
    part=deepcopy(data['colliders'][0]);part.update(id='opposing_finger',body='link8');data['colliders'].append(part)
    w.scene=CollisionScene(data);p=PhysicalContact(w,roles=['support_arm0'])
    key=frozenset(('finger','opposing_finger'))
    assert not p.keys
    assert p.public()['native_auxiliary_interactions']['auxiliary_grasp_inference'] is False
    p.begin([[-.3]])  # Initial deep contact can only separate, never deepen.
    assert p.check([[-.3]])['clear']
    r['control_owner']='target';w.scene=CollisionScene(data)
    assert key not in PhysicalContact(w,roles=['left']).keys


def test_contact_depth_diagnostics_remain_operator_only(caplog):
    p=PhysicalContact(world(),roles=['left']);p.begin([[-.3]])
    result=p.check([[-.098]])
    assert result['reason']=='physical_contact_penetration_limit'
    assert 'OPENETA_CONTACT_BOUND' in caplog.text and 'depth_bound_m' in caplog.text
    public=json.dumps(result)
    assert 'depth_bound_m' not in public and 'geometry_sources' not in public
    assert p.check([[-.0995]])['clear']
