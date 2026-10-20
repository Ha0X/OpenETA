from types import SimpleNamespace
from itertools import product
import numpy as np
import pytest
from test_robodojo_attached_geometry import grasp_scene
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.support_contact import SupportPlacement, SupportContactScene


def setup():
    world,grants=grasp_scene()
    snapshot=world.scene.snapshot
    snapshot['colliders'][2]['vertices']=list(product(*zip([-1.1,-.03,-.03],[-.9,.03,.03])))
    snapshot['colliders'][3].update(id='table',vertices=list(product(*zip([-2,-1,-.10],[0,1,-.04]))))
    world.scene=CollisionScene(snapshot);world.table_id='table'
    world.refresh=lambda:None;world.positions=lambda:[np.array([-1.,0.,0.])]
    pose=np.eye(4);pose[0,3]=-1.
    c=SimpleNamespace(world=world,contact_grants=grants,roles=['left'],tcp=lambda i:pose.copy())
    target=pose.copy();target[2,3]=-.04
    return c,{'left':[target]}


def test_explicit_placement_clips_robot_target_and_only_allows_held_table_pair():
    c,targets=setup();p=SupportPlacement(c,'left',[-1,0,-.04],targets)
    assert abs(p.effective[2,3]+.0102)<1e-8
    assert p.pairs==[('held','table')]
    assert p.public()['gripper_release']=='agent_decision'
    scene=SupportContactScene(c.world.scene,p)
    assert scene.check(c.world.positions(),allowed_pairs=c.contact_grants['left']['pairs'])['clear']
    t=np.eye(4);t[2,3]=-.0105
    c.world.scene.world_transforms[2]=t;c.world.scene.set_transform(2,t)
    predicted = p.check(c.world.scene,c.world.positions())
    assert predicted['status']=='clear'
    assert predicted['measurement']['evaluation'] == 'predicted_geometry'
    assert not p.reached()  # A predicted contact must not become measured evidence.
    p.check(c.world.scene,c.world.positions(),measured=True)
    assert p.reached()
    assert p.public()['measurement']['evaluation'] == 'measured_geometry'
    t[2,3]=-.012
    c.world.scene.world_transforms[2]=t;c.world.scene.set_transform(2,t)
    assert p.check(c.world.scene,c.world.positions())['reason']=='support_penetration_limit'


def test_contact_requires_actual_grasp_and_local_table_mark():
    c,targets=setup()
    with pytest.raises(ValueError,match='table top'):SupportPlacement(c,'left',[-1,0,.1],targets)
    with pytest.raises(ValueError,match='support patch'):SupportPlacement(c,'left',[-.3,0,-.04],targets)
    c.contact_grants={}
    with pytest.raises(ValueError,match='two-finger'):SupportPlacement(c,'left',[-1,0,-.04],targets)


def test_positive_support_gap_reports_zero_penetration():
    c, targets = setup()
    p = SupportPlacement(c, 'left', [-1,0,-.04], targets)
    result = p.check(c.world.scene, c.world.positions(), measured=True)
    assert result['measurement']['signed_clearance_m'] > 0
    assert result['measurement']['maximum_penetration_m'] == 0
    assert result['status'] == 'clear'


def test_contact_does_not_allow_arbitrary_transport_or_rotation():
    c,targets=setup();targets['left'][0][0,3]+=.1
    with pytest.raises(ValueError,match='unchanged'):SupportPlacement(c,'left',[-1,0,-.04],targets)
    c,targets=setup();targets['left'].append(targets['left'][0])
    with pytest.raises(ValueError,match='one short target'):SupportPlacement(c,'left',[-1,0,-.04],targets)


def test_rounded_zero_height_is_reported_and_never_executes_upward():
    c, targets = setup()
    targets['left'][0][2, 3] = .0000004713
    p = SupportPlacement(c, 'left', [-1, 0, -.04], targets)
    assert p.effective[2, 3] == c.tcp(0)[2, 3] == 0.
    assert p.requested[2, 3] == .0000004713
    assert p.public()['target_normalization'] == {
        'kind': 'near_zero_support_height', 'requested_delta_z_m': .0000004713,
        'effective_delta_z_m': 0., 'rounding_tolerance_m': 1e-6}
    for height in [1.0001e-6, .001, -.0500001]:
        targets['left'][0][2, 3] = height
        with pytest.raises(ValueError, match='requested delta_z_m='):
            SupportPlacement(c, 'left', [-1, 0, -.04], targets)


def test_real_downward_support_motion_is_not_normalized():
    c, targets = setup()
    p = SupportPlacement(c, 'left', [-1, 0, -.04], targets)
    assert p.public()['target_normalization'] is None
    assert p.effective[2, 3] < 0.


def test_exact_five_centimetres_after_public_pose_subtraction_is_admitted():
    c, targets = setup()
    current = c.tcp(0); current[2,3] = .905471404662492
    c.tcp = lambda _:current.copy()
    targets['left'][0][2,3] = current[2,3] - .05
    assert targets['left'][0][2,3] - current[2,3] < -.05
    p = SupportPlacement(c, 'left', [-1,0,-.04], targets)
    adjustment = p.public()['target_normalization']
    assert adjustment['kind'] == 'support_descent_boundary_rounding'
    assert adjustment['effective_delta_z_m'] == -.05
    # The existing earlier support stop still takes precedence.
    assert p.effective[2,3] >= current[2,3] - .05
    targets['left'][0][2,3] = current[2,3] - .05000000001
    with pytest.raises(ValueError, match='requested delta_z_m='):
        SupportPlacement(c, 'left', [-1,0,-.04], targets)


def test_patch_expansion_requires_explicit_bounded_radius_and_keeps_depth_limit():
    c,targets=setup()
    with pytest.raises(ValueError,match='required radius'):
        SupportPlacement(c,'left',[-1,.2,-.04],targets)
    p=SupportPlacement(c,'left',[-1,.2,-.04],targets,patch_radius_m=.30)
    assert p.patch_radius_m==.30 and p.max_penetration_m==.001
    for invalid in [True,.04,.351,float('nan')]:
        with pytest.raises(ValueError,match='radius'):
            SupportPlacement(c,'left',[-1,0,-.04],targets,patch_radius_m=invalid)


def test_predicted_table_bound_avoids_distance_queries_but_measurement_keeps_fcl(monkeypatch):
    c,targets=setup();p=SupportPlacement(c,'left',[-1,0,-.04],targets)
    scene=c.world.scene;original=scene.signed_distances;calls=[]
    def measured(*args,**kwargs):
        calls.append(kwargs)
        return original(*args,**kwargs)
    monkeypatch.setattr(scene,'signed_distances',measured)
    transform=np.eye(4);transform[2,3]=-.0105
    scene.world_transforms[2]=transform;scene.set_transform(2,transform)
    result=p.check(scene,c.world.positions())
    assert result['status']=='clear' and not calls
    assert result['measurement']['distance_method']=='calibrated_table_halfspace_separation_upper_bound'
    assert np.isclose(result['measurement']['maximum_penetration_m'],.0005)
    assert not p.reached()
    result=p.check(scene,c.world.positions(),measured=True)
    assert calls==[{'contact_bounds':True}]
    assert result['measurement']['distance_method']=='fcl_signed_distance' and p.reached()
    transform[2,3]=-.0111;scene.world_transforms[2]=transform;scene.set_transform(2,transform)
    assert p.check(scene,c.world.positions())['status']=='blocked'
    assert len(calls)==1  # The bound rejects excessive depth without FCL as well.
