import numpy as np
from types import SimpleNamespace as NS
from sim.envs.robodojo.cloth_execution import ClothExecution
from sim.envs.robodojo.deformable_collision import ClothCollisionSnapshot
from sim.envs.robodojo.deformable_surface import MaterialSurface
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_collision_geometry import fixture
from test_robodojo_deformable_surface import sheet


def setup():
    data=fixture(start=-.058188);r=data['robots'][0];r['role']='left'
    r['body_frames']['link7']=r['body_frames'].pop('tool');r['joints'][0]['child']='link7'
    data['colliders'][0]['body']='link7'
    scene=CollisionScene(data);p,f=sheet();p=p[:,[2,0,1]]
    surface=MaterialSurface(p,f,contact_offset_m=.008)
    c=ClothExecution.__new__(ClothExecution)
    c.control=NS(world=NS(scene=scene,table_id='world'));c.snapshots=[ClothCollisionSnapshot(surface.frame)]
    c.recovery_events=[]
    return c,surface


def test_positive_gap_can_separate_but_closing_or_large_command_rejected():
    c,_=setup();start=[np.array([-.058188])];target=[np.array([-.0585])]
    assert c.check_segment(start,target)['clear']
    measured=c.check_after_physics(target)
    assert measured['clear'] and c.recovery_events[-1]['normal_margin_restored']
    assert not c.check_segment(start,[np.array([-.0581])])['clear']
    assert not c.check_segment(start,[np.array([-.1])])['clear']


def test_initial_negative_gap_does_not_get_exit_exception():
    c,_=setup()
    assert not c.check_segment([np.array([-.057])],[np.array([-.059])])['clear']


def test_new_deformation_worsening_after_physics_stops_recovery():
    c,s=setup();start=[np.array([-.058188])];target=[np.array([-.05819])]
    assert c.check_segment(start,target)['clear']
    p=s.frame.vertices.copy();p[:,0]-=.0001
    c.snapshots=[ClothCollisionSnapshot(s.update(p,simulation_time_s=.004))]
    result=c.check_after_physics(target)
    assert not result['clear'] and result['reason']=='cloth_recovery_measured_worsening_or_new_contact'


def test_recovery_physics_budget_is_bounded():
    c,_=setup();c.recovery_events=[{}]*40
    result=c.check_segment([np.array([-.058188])],[np.array([-.0585])])
    assert not result['clear'] and result['reason']=='cloth_recovery_physics_budget'


def test_separating_old_finger_does_not_authorize_new_other_hand_contact():
    from copy import deepcopy
    c,s=setup();data=deepcopy(c.control.world.scene.snapshot)
    other=deepcopy(data['robots'][0]);other['role']='right'
    other['joint_positions']=[-.0585];other['body_frames']['link7'][0][3]=-.0585
    data['robots'].append(other)
    collider=deepcopy(data['colliders'][0]);collider.update(id='right_finger',robot_index=1)
    data['colliders'].append(collider)
    c.control.world.scene=CollisionScene(data)
    start=[np.array([-.058188]),np.array([-.0585])]
    target=[np.array([-.0585]),np.array([-.0581])]
    assert not c.check_segment(start,target)['clear']
