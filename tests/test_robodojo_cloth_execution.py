from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation

from sim.envs.robodojo.cloth_execution import ClothExecution
from sim.envs.robodojo.deformable_collision import ClothCollisionSnapshot
from sim.envs.robodojo.deformable_surface import MaterialSurface
from test_robodojo_deformable_surface import sheet
from test_robodojo_collision_geometry import fixture
from sim.envs.robodojo.collision_geometry import CollisionScene


def coordinator(grants=None):
    c=ClothExecution.__new__(ClothExecution)
    c.control=NS(roles=['left','right'],tcp=lambda index:np.eye(4),contact_grants=grants or {},physics_steps=12)
    c.evidence={}
    return c


def pose(x, angle=0):
    p=np.eye(4);p[0,3]=x;p[:3,:3]=Rotation.from_euler('z',angle).as_matrix();return p


def test_short_segment_uses_common_fraction_and_reports_unexecuted_waypoints():
    c=coordinator();targets={'left':[pose(.4),pose(.5)],'right':[pose(.2)]}
    effective,mode=c.prepare_move(targets,'synchronized')
    assert mode=='synchronized'
    assert effective['left'][0][0,3]==.12 and effective['right'][0][0,3]==.06
    assert targets['left'][0][0,3]==.4 and len(targets['left'])==2
    receipt=c.decorate({'reached_target':True},'move')
    assert not receipt['reached_target'] and receipt['reached_execution_target']
    assert receipt['continuation_required'] and receipt['host_adjustments']


def test_shared_cloth_cannot_be_retimed_or_serialized():
    grants={r:{'kind':'cloth_patch','cloth_index':0} for r in ('left','right')}
    c=coordinator(grants)
    effective,mode=c.prepare_move({'left':[pose(.2)],'right':[pose(-.2)]},'independent')
    assert mode=='synchronized' and np.isclose(effective['left'][0][0,3],.05)
    assert any(e['kind']=='shared_cloth_synchronized' for e in c.events)


def test_different_materials_do_not_force_shared_synchronization():
    c=coordinator({r:{'kind':'cloth_patch','cloth_index':i} for i,r in enumerate(('left','right'))})
    _,mode=c.prepare_move({'left':[pose(.01)],'right':[pose(.01)]},'independent')
    assert mode=='independent' and not c.truncated


def test_rotation_is_bounded_and_small_original_target_can_be_completed():
    c=coordinator();result,_=c.prepare_move({'left':[pose(0,2.)]},'synchronized')
    assert np.isclose(Rotation.from_matrix(result['left'][0][:3,:3]).magnitude(),.5)
    c.prepare_move({'left':[pose(.01)]},'synchronized')
    receipt=c.decorate({'reached_target':True},'move')
    assert receipt['reached_target'] and 'continuation_required' not in receipt


def test_swept_robot_certificate_cannot_cross_stationary_cloth():
    data=fixture();data['robots'][0]['role']='left';scene=CollisionScene(data)
    p,f=sheet();p=p[:,[2,0,1]] # vertical sheet across the robot slide axis
    material=MaterialSurface(p,f,contact_offset_m=.008)
    c=coordinator();c.control.world=NS(scene=scene,table_id='world')
    c.snapshots=[ClothCollisionSnapshot(material.frame)]
    assert c.check([np.array([-1.])])['clear']
    result=c.check_segment([np.array([-1.])],[np.array([1.])])
    assert not result['clear']
    assert result['reason'] in ('cloth_snapshot_clearance','cloth_snapshot_check_budget','cloth_snapshot_resolution','predicted_cloth_clearance_at_command_target')
    assert c.check_segment([np.array([-1.])],[np.array([-.8])])['clear']


def test_stationary_other_arm_near_cloth_does_not_inherit_moving_arm_bound():
    from copy import deepcopy
    data=fixture();data['robots'][0]['role']='left'
    other=fixture(start=-.0582006)['robots'][0];other['role']='right';data['robots'].append(other)
    item=deepcopy(data['colliders'][0]);item.update(id='right_finger',robot_index=1);data['colliders'].append(item)
    scene=CollisionScene(data);p,f=sheet();p=p[:,[2,0,1]]
    surface=MaterialSurface(p,f,contact_offset_m=.008)
    c=coordinator();c.control.world=NS(scene=scene,table_id='world');c.snapshots=[ClothCollisionSnapshot(surface.frame)]
    start=[np.array([-1.]),np.array([-.0582006])]
    target=[np.array([-.8]),np.array([-.0582006])]
    assert c.check(start)['clear'] and c.check(target)['clear']
    result=c.check_segment(start,target)
    assert result['clear'] and result['samples']==1
    # Moving that same near hand into the cloth must still be rejected.
    target[1]=np.array([-.04])
    assert not c.check_segment(start,target)['clear']


def test_blocked_joint_target_reports_predicted_part_before_recursive_budget():
    data=fixture();data['robots'][0]['role']='left';scene=CollisionScene(data)
    p,f=sheet();p=p[:,[2,0,1]];surface=MaterialSurface(p,f,contact_offset_m=.008)
    c=coordinator();c.control.world=NS(scene=scene,table_id='world');c.snapshots=[ClothCollisionSnapshot(surface.frame)]
    result=c.check_segment([np.array([-1.])],[np.array([0.])])
    assert result['reason']=='predicted_cloth_clearance_at_command_target'
    assert result['samples']==1 and 'not_measured_contact' in result['evidence']
    assert result['contacts'][0]['authorization_scope']=='unapproved_robot_cloth_contact'


def test_half_open_approach_does_not_arm_holding_loss_but_closed_loss_stops():
    c=coordinator();c.seen=set();c.lost_steps={};width=[.45];stops=[]
    c.control._grippers=lambda:width;c.control.native=NS(sim=NS(physics_dt=.004))
    c.control.operation='move';c.control._halt=lambda *args:stops.append(args)
    near=lambda:{'proximity':'both_fingers_near_patch'}
    far=lambda:{'proximity':'two_finger_proximity_not_observed'}
    c._update_contact_evidence('left',near())
    for _ in range(20):c._update_contact_evidence('left',far())
    assert not stops and not c.seen
    width[0]=0.;report=near();c._update_contact_evidence('left',report)
    assert report['holding_loss_monitor_armed']
    for _ in range(10):c._update_contact_evidence('left',far())
    assert stops[-1][0]=='cloth_local_contact_evidence_lost'
    stops.clear();width[0]=.4;c._update_contact_evidence('left',far())
    assert not c.seen and not stops
