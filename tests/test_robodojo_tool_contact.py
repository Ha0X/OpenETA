from itertools import product
import numpy as np
import pytest

from test_robodojo_support_contact import setup
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.tool_contact import ToolStroke, marked_dynamic_body
from sim.envs.robodojo.support_contact import SupportContactScene


def reorient_setup():
    from scipy.spatial.transform import Rotation
    from sim.envs.robodojo.scene_geometry import forward_kinematics
    from sim.envs.robodojo.x5_calibration import link_to_tcp
    c, targets = stroke_setup()
    current = link_to_tcp(forward_kinematics(c.world.scene.snapshot['robots'][0], c.world.positions()[0])['link6'])
    c.tcp = lambda _: current.copy()
    targets['left'][0] = current.copy()
    targets['left'][0][1,3] += .02
    targets['left'][0][:3,:3] = Rotation.from_euler('z', 12, degrees=True).as_matrix() @ current[:3,:3]
    return c, targets


def test_reorientation_is_explicit_and_does_not_clip_the_requested_pose():
    c, targets = reorient_setup()
    with pytest.raises(ValueError, match='unchanged orientation'):
        ToolStroke(c, 'left', [-1,0,-.04], [], targets)
    p = ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode='reorient')
    assert np.array_equal(p.effective, targets['left'][0])
    assert p.public()['mode'] == 'tool_reorient'
    assert p.public()['orientation_reference'] == 'shortest_commanded_arc'
    assert p.public()['max_stroke_m'] == .03
    assert p.pairs == [('held', 'table')]
    assert not p.reached()


@pytest.mark.parametrize('mode', ['adjust', 'reorient'])
def test_contact_adjustment_admits_vertical_motion_without_requiring_rotation(mode):
    c, targets = reorient_setup()
    targets['left'][0] = c.tcp(0)
    targets['left'][0][2,3] += .02
    with pytest.raises(ValueError, match='3mm height'):
        ToolStroke(c, 'left', [-1,0,-.04], [], targets)
    p = ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode=mode)
    assert np.array_equal(p.effective, targets['left'][0])
    assert p.public()['mode'] == 'tool_'+mode
    assert p.check(c.world.scene,c.world.positions(),measured=True)['status']=='clear'


@pytest.mark.parametrize('mode', ['adjust', 'reorient'])
@pytest.mark.parametrize('translation,angle', [(.031, 12), (.02, 15.1), (.02, 90)])
def test_reorientation_refuses_out_of_scope_pose(translation, angle, mode):
    from scipy.spatial.transform import Rotation
    c, targets = reorient_setup()
    targets['left'][0][1,3] = translation
    targets['left'][0][:3,:3] = Rotation.from_euler('z', angle, degrees=True).as_matrix() @ c.tcp(0)[:3,:3]
    with pytest.raises(ValueError, match='3cm TCP translation and 15 degrees'):
        ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode=mode)


@pytest.mark.parametrize('measured', [False, True])
def test_reorientation_envelope_accepts_arc_and_rejects_overshoot_and_wrong_axis(monkeypatch, measured):
    from scipy.spatial.transform import Rotation
    from sim.envs.robodojo import x5_calibration
    c, targets = reorient_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode='reorient')
    pose = p.reference_tcp.copy()
    pose[1,3] += .01
    # Isolate the TCP corridor; the existing fixture still independently runs
    # table penetration/patch checks from its unchanged actual colliders.
    monkeypatch.setattr(x5_calibration, 'link_to_tcp', lambda _: pose.copy())
    for degrees in (0, 6, 12):
        pose[:3,:3] = Rotation.from_euler('z', degrees, degrees=True).as_matrix() @ p.reference_tcp[:3,:3]
        assert p.check(c.world.scene, c.world.positions(), measured=measured)['status'] == 'clear'
    for axis, degrees in [('z', 13), ('z', -1), ('x', 2)]:
        pose[:3,:3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix() @ p.reference_tcp[:3,:3]
        result = p.check(c.world.scene, c.world.positions(), measured=measured)
        assert result['reason'] == 'tool_stroke_path_envelope_exceeded'
        assert result['contact_mode'] == 'reorient'
    pose[:3,:3] = p.reference_tcp[:3,:3];pose[0,3] += .004
    assert p.check(c.world.scene, c.world.positions(), measured=measured)['path_error_m'] == pytest.approx(.004)


def test_reorientation_keeps_table_penetration_and_unmarked_body_checks():
    c, targets = reorient_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode='reorient')
    transform = np.eye(4);transform[2,3] = -.012
    c.world.scene.world_transforms[2] = transform;c.world.scene.set_transform(2, transform)
    assert p.check(c.world.scene, c.world.positions(), measured=True)['reason'] == 'support_penetration_limit'
    c, targets = reorient_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [], targets, mode='reorient')
    transform = np.eye(4);transform[1,3] = -.08
    c.world.scene.world_transforms[4] = transform;c.world.scene.set_transform(4,transform)
    from sim.envs.robodojo.attached_geometry import attached_scene
    scene, _ = attached_scene(c.world,c.world.positions(),c.contact_grants)
    checked = SupportContactScene(scene,p).check(c.world.positions(),allowed_pairs=c.contact_grants['left']['pairs'])
    assert not checked['clear']


def stroke_setup():
    control, targets = setup()
    snapshot = control.world.scene.snapshot
    snapshot['colliders'].append({'id':'workpiece','robot_index':None,'body':None,
        'approximation':'convexHull','vertices':list(product(*zip([-1.01,.08,-.03],[-.99,.10,-.01]))),
        'triangles':[]})
    control.world.scene = CollisionScene(snapshot)
    control.world.dynamic = {'body':[2], 'workpiece':[4]}
    targets['left'][0][2,3] = 0.
    targets['left'][0][1,3] = .04
    return control, targets


def test_stroke_requires_explicit_unambiguous_body_and_bounded_pose():
    c, targets = stroke_setup()
    p = ToolStroke(c,'left',[-1,0,-.04],[[-1,.09,-.02]],targets)
    assert p.pairs == [('held','table'),('held','workpiece'),('workpiece','table')]
    assert p.public()['workpiece_count'] == 1
    assert not p.reached()
    assert np.array_equal(p.effective,targets['left'][0])
    with pytest.raises(ValueError,match='differ'):
        ToolStroke(c,'left',[-1,0,-.04],[[-1,0,0]],targets)
    with pytest.raises(ValueError,match='exactly one'):
        marked_dynamic_body(c.world,[-1,.5,0])
    targets['left'][0][1,3] = .06
    with pytest.raises(ValueError,match='5cm'):
        ToolStroke(c,'left',[-1,0,-.04],[[-1,.09,-.02]],targets)


def test_actual_workpiece_penetration_and_patch_are_monitored():
    c, targets = stroke_setup()
    p = ToolStroke(c,'left',[-1,0,-.04],[[-1,.09,-.02]],targets)
    scene = c.world.scene
    t = np.eye(4); t[1,3] = -.0505
    scene.world_transforms[4] = t; scene.set_transform(4,t)
    assert p.check(scene,c.world.positions(),measured=True)['status']=='clear'
    t[1,3] = -.052
    scene.world_transforms[4] = t; scene.set_transform(4,t)
    assert p.check(scene,c.world.positions(),measured=True)['reason']=='tool_workpiece_penetration_limit'
    # Full path planning does not pretend static target penetration models pushing.
    assert p.check(scene,c.world.positions(),measured=False)['status']=='clear'
    t[1,3] = .2
    scene.world_transforms[4] = t; scene.set_transform(4,t)
    assert p.check(scene,c.world.positions(),measured=True)['reason']=='tool_workpiece_patch_exceeded'


def test_explicit_tool_pair_never_authorizes_fingers_to_contact_workpiece():
    c, targets = stroke_setup()
    p = ToolStroke(c,'left',[-1,0,-.04],[[-1,.09,-.02]],targets)
    t = np.eye(4);t[1,3] = -.04
    c.world.scene.world_transforms[4] = t;c.world.scene.set_transform(4,t)
    scene = SupportContactScene(c.world.scene,p)
    checked = scene.check(c.world.positions(),allowed_pairs=c.contact_grants['left']['pairs'])
    assert not checked['clear']
    assert any('finger7' in (x['a'],x['b']) and 'workpiece' in (x['a'],x['b'])
               for x in checked['contacts'])


def test_tool_stroke_rejects_spatial_detour_outside_agent_segment():
    c,targets=stroke_setup()
    p=ToolStroke(c,'left',[-1,0,-.04],[[-1,.09,-.02]],targets)
    q=c.world.positions();q[0][0]+=.004
    assert p.check(c.world.scene,q)['reason']=='tool_stroke_path_envelope_exceeded'


def test_workpiece_feedback_names_opaque_pairs_and_preserves_per_pair_distance_basis():
    c, targets = stroke_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [[-1,.09,-.02]], targets)
    scene = c.world.scene
    exact = frozenset(('held','workpiece'))
    bounded = frozenset(('workpiece','table'))
    scene.last_contact_distance_method = 'conservative_projection_bound_on_fcl_disagreement'
    scene.last_contact_distance_methods = {
        exact: 'fcl_signed_distance', bounded: scene.last_contact_distance_method}
    feedback = p._workpiece_pair_feedback(scene, {exact:-.002, bounded:-.0001})
    first, second = feedback['limiting_pairs']
    assert first['geometry_a']=='geometry_2' and first['geometry_b']=='geometry_4'
    assert first['participants'][0]['contact_authorizations'][0]['role']=='left'
    assert first['penetration_estimate']=='signed_geometry_distance'
    assert second['penetration_estimate']=='conservative_upper_bound'
    assert second['participants'][0]=={'kind':'static_support','part':'table'}
    assert feedback['checked_pair_count']==2


def test_mark_rejection_distinguishes_no_match_from_ambiguous_contact():
    c, _ = stroke_setup()
    with pytest.raises(ValueError,match='no_body_match, matched_body_count=0'):
        marked_dynamic_body(c.world,[-1,.5,0])
    c.world.dynamic['another_body'] = c.world.dynamic['workpiece']
    with pytest.raises(ValueError,match='ambiguous_body_match, matched_body_count=2'):
        marked_dynamic_body(c.world,[-1,.09,-.02])


def test_near_working_pairs_survive_many_decomposed_table_contacts_without_new_queries(monkeypatch):
    c, targets = stroke_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [[-1,.09,-.02]], targets)
    scene = c.world.scene
    for number in range(3):
        scene.items.append({**scene.items[4], 'id': f'workpiece_part_{number}'})
    p.workpieces['workpiece'] = (4, 5, 6, 7)
    distances = {frozenset((scene.items[i]['id'], 'table')): -.0008 + (i-4)*.0001
                 for i in range(4, 8)}
    working_pair = frozenset(('held', 'workpiece'))
    distances[working_pair] = .0005
    scene.last_contact_distance_method = 'fcl_signed_distance'
    scene.last_contact_distance_methods = {pair: 'fcl_signed_distance' for pair in distances}
    monkeypatch.setattr(scene, 'signed_distances', lambda *a, **k: pytest.fail('feedback must reuse measured distances'))
    feedback = p._workpiece_pair_feedback(scene, distances)
    assert all(any(x['kind']=='static_support' for x in row['participants'])
               for row in feedback['limiting_pairs'])
    near, = feedback['near_interaction_pairs']
    assert near['pair_scope']=='held_tool_to_workpiece'
    assert near['tool_role']=='left' and near['workpiece_reference_indices']==[0]
    assert near['signed_clearance_m']==.0005 and near['maximum_penetration_m']==0
    assert len(feedback['near_workpiece_table_pairs'])==1
    assert not feedback['force_or_actual_contact_inferred']


def test_near_pair_feedback_preserves_bounds_and_omits_distant_geometry():
    c, targets = stroke_setup()
    p = ToolStroke(c, 'left', [-1,0,-.04], [[-1,.09,-.02]], targets)
    scene = c.world.scene
    pair = frozenset(('held','workpiece'))
    scene.last_contact_distance_method = 'aabb_separation_lower_bound'
    scene.last_contact_distance_methods = {pair: 'aabb_separation_lower_bound'}
    feedback = p._workpiece_pair_feedback(scene, {pair: .0001})
    near, = feedback['near_interaction_pairs']
    assert near['distance_method']=='aabb_separation_lower_bound'
    assert near['penetration_estimate']=='no_overlap_certified_by_enclosing_boxes'
    assert 'overstate proximity' in feedback['near_pair_basis']
    assert not feedback['force_or_actual_contact_inferred']
    assert p._workpiece_pair_feedback(scene, {pair: .003})['near_interaction_pairs']==[]


def test_table_only_slide_keeps_unmarked_body_collisions():
    from sim.envs.robodojo.attached_geometry import attached_scene
    c,targets=stroke_setup()
    p=ToolStroke(c,'left',[-1,0,-.04],[],targets)
    assert p.pairs==[('held','table')] and p.workpieces=={}
    assert p.public()['workpiece_count']==0
    assert p.check(c.world.scene,c.world.positions(),measured=True)['status']=='clear'
    transform=np.eye(4);transform[1,3]=-.08
    c.world.scene.world_transforms[4]=transform;c.world.scene.set_transform(4,transform)
    scene,_=attached_scene(c.world,c.world.positions(),c.contact_grants)
    checked=SupportContactScene(scene,p).check(c.world.positions(),allowed_pairs=c.contact_grants['left']['pairs'])
    assert not checked['clear']
    assert any(set((contact['a'],contact['b']))=={'held','workpiece'} for contact in checked['contacts'])


@pytest.mark.parametrize('measured',[False,True])
def test_path_envelope_feedback_reports_actual_error_and_evaluation_kind(measured):
    c,targets=stroke_setup();p=ToolStroke(c,'left',[-1,0,-.04],[],targets)
    q=c.world.positions();q[0][0]+=.004
    result=p.check(c.world.scene,q,measured=measured)
    assert result['reason']=='tool_stroke_path_envelope_exceeded'
    assert result['path_error_m']==pytest.approx(.004)
    assert result['path_limit_m']==.003
    assert result['orientation_error_rad']==0.
    assert result['orientation_limit_rad']==.01
    assert result['evaluation']==('measured_geometry' if measured else 'predicted_geometry')
