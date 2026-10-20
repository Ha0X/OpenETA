import numpy as np
import pytest
fcl=pytest.importorskip('fcl')

from sim.envs.robodojo.deformable_collision import ClothCollisionSnapshot
from sim.envs.robodojo.deformable_surface import MaterialSurface, DeformableCoverageError
from test_robodojo_deformable_surface import sheet


def make():
    p,f=sheet()
    return MaterialSurface(p,f,contact_offset_m=.008)


def probe(x,y,z,r=.001):
    return fcl.CollisionObject(fcl.Sphere(r),fcl.Transform(np.array([x,y,z])))


def test_collision_uses_live_deformation_and_preserves_old_snapshot():
    s=make(); a=ClothCollisionSnapshot(s.frame)
    point=probe(.04,.04,.02)
    assert a.clearance(point)>0
    p=s.frame.vertices.copy();p[:,2]=.02
    b=ClothCollisionSnapshot(s.update(p,simulation_time_s=.004))
    assert b.clearance(point)<0
    assert a.clearance(point)>0


def test_native_thickness_is_not_a_zero_thickness_triangle_test():
    s=make(); collision=ClothCollisionSnapshot(s.frame)
    assert collision.clearance(probe(.04,.04,.005))<0
    assert collision.clearance(probe(.04,.04,.02))==pytest.approx(.011,abs=1e-7)


def test_contact_patch_authorizes_only_own_fingers_and_local_material():
    s=make(); patch=s.material_patch([.04,.04,0],radius_m=.025)
    collision=ClothCollisionSnapshot(s.frame,patches={'left':patch})
    center=probe(.04,.04,0.)
    assert collision.clearance(center,role='left',finger=True)>0
    assert collision.clearance(center,role='right',finger=True)<0
    assert collision.clearance(center,role='left',finger=False)<0
    assert collision.clearance(probe(.0,.0,0.),role='left',finger=True)<0


def test_authorization_does_not_expand_after_far_material_folds_into_hand():
    s=make(); patch=s.material_patch([0.,0.,0.],radius_m=.025)
    p=s.frame.vertices.copy();p[-9:]=p[:9]+[0,0,.001]
    snapshot=ClothCollisionSnapshot(s.update(p,simulation_time_s=.004),patches={'left':patch})
    assert snapshot.clearance(probe(.0,.0,.001),role='left',finger=True)<0


def test_new_material_cannot_reuse_previous_contact_authorization():
    s=make(); patch=s.material_patch([0,0,0])
    with pytest.raises(DeformableCoverageError):ClothCollisionSnapshot(make().frame,patches={'left':patch})


def test_explicit_stack_contact_keeps_other_parts_and_later_layers_checked():
    p,f=sheet();points=np.concatenate([p,p+[0,0,-.005],p+[0,0,-.03]])
    faces=np.concatenate([f,f+len(p),f+2*len(p)])
    s=MaterialSurface(points,faces,contact_offset_m=.008)
    patch=s.local_layers([.04,.04,0],[0,0,-1])
    snap=ClothCollisionSnapshot(s.frame,patches={'left':patch});center=probe(.04,.04,-.002)
    assert snap.clearance(center,role='left',finger=True)>0
    assert snap.clearance(center,role='right',finger=True)<0
    assert snap.clearance(center,role='left',finger=False)<0
    points[2*len(p):,2]=-.001
    snap=ClothCollisionSnapshot(s.update(points,simulation_time_s=.004),patches={'left':patch})
    assert snap.clearance(center,role='left',finger=True)<0


def test_broad_phase_never_claims_more_clearance_than_exact_query():
    from test_robodojo_collision_geometry import fixture
    from sim.envs.robodojo.collision_geometry import CollisionScene
    data=fixture();data['robots'][0]['role']='left'
    rigid=CollisionScene(data);s=make();c=ClothCollisionSnapshot(s.frame)
    for position in [-1.,-.2,-.06,-.04,0.,.04,.2]:
        r=c.check(rigid,[[position]])
        exact=c.clearance(rigid.objects[0],role='left')
        assert r['clearance_lower_bound_m']<=exact+1e-8
        assert r['clear']==(exact>.0002)
        assert r['future_deformation_predicted'] is False
    assert c.check(rigid,[[-1.]])['queries']==0


def test_world_cloth_support_exception_is_only_the_selected_table():
    from copy import deepcopy
    from test_robodojo_collision_geometry import fixture
    from sim.envs.robodojo.collision_geometry import CollisionScene
    data=fixture();data['robots'][0]['role']='left'
    scene=CollisionScene(data);c=ClothCollisionSnapshot(make().frame)
    assert not c.check(scene,[[-1.]],allowed_world_ids=[])['clear']
    assert c.check(scene,[[-1.]],allowed_world_ids=['world'])['clear']
    extra=deepcopy(data['colliders'][1]);extra['id']='other_obstacle';data['colliders'].append(extra)
    assert not c.check(CollisionScene(data),[[-1.]],allowed_world_ids=['world'])['clear']
