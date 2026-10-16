from copy import deepcopy
from types import SimpleNamespace
import numpy as np

from sim.envs.robodojo.clearance_recovery import ClearanceRecovery, near_clearance_pairs
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_collision_geometry import fixture


def make(start=-.100186):
    data=fixture(start=start)
    robot=data['robots'][0];robot['role']='left'
    robot['body_frames']['link6']=robot['body_frames'].pop('tool')
    robot['joints'][0]['child']='link6';data['colliders'][0]['body']='link6'
    return data


def test_positive_gap_can_retreat_without_disabling_geometric_collision_checks():
    scene=CollisionScene(make());start=[[-.100186]]
    assert scene.check_path(start,[[-.12]])['reason']=='insufficient_clearance'
    pairs,_=near_clearance_pairs(scene,start)
    recovery=ClearanceRecovery(scene,start,pairs)
    assert recovery.check_segment(start,[[-.12]])['status']=='clear'
    assert recovery.check_segment(start,[[-.1001]])['reason']=='positive_gap_would_shrink'
    assert recovery.check_segment(start,[[.12]])['status']!='clear'
    assert recovery.check_actual([[-.101]])
    assert recovery.diagnostics()['actual_gap_must_remain_positive']


def test_submicron_positive_gap_retreat_refines_until_nonintersection_is_proved():
    scene=CollisionScene(make(-.10000012));start=[[-.10000012]]
    pairs,_=near_clearance_pairs(scene,start)
    recovery=ClearanceRecovery(scene,start,pairs)
    assert recovery.check_segment(start,[[-.12]])['status']=='clear'
    # Ordinary padding, inward travel, and finite compute budgets still apply.
    assert scene.check_path(start,[[-.12]])['status']=='blocked'
    assert scene.check_path(start,[[-.099]],margin_m=0.)['status']!='clear'
    assert scene.check_path(start,[[-.12]],margin_m=0.,max_nodes=3)['status']=='unknown'
    assert not recovery.check_actual([[-.099999999]])


def vertical_scene():
    value=.10000012
    data=make(value);robot=data['robots'][0]
    robot['joints'][0]['axis']='Z'
    robot['body_frames']['link6'][0][3]=0.
    robot['body_frames']['link6'][2][3]=value
    return data


def test_table_plane_proof_admits_submicron_lift_but_never_masks_new_obstacles():
    data=vertical_scene();scene=CollisionScene(data);start=[[.10000012]]
    pairs,_=near_clearance_pairs(scene,start)
    recovery=ClearanceRecovery(scene,start,pairs,table_id='world')
    result=recovery.check_segment(start,[[.12]])
    assert result['status']=='clear' and result['separating_table_plane_pair_count']==1
    assert recovery.diagnostics()['separating_table_plane_certificates']==1
    assert recovery.check_segment(start,[[.099]])['status']!='clear'
    assert not recovery.check_actual([[.099999999]])
    other=deepcopy(data['colliders'][1]);other['id']='upper_obstacle'
    other['vertices']=(np.array(other['vertices'])+[0.,0.,.24]).tolist()
    data['colliders'].append(other);scene=CollisionScene(data)
    recovery=ClearanceRecovery(scene,start,pairs,table_id='world')
    assert recovery.check_segment(start,[[.15]])['status']!='clear'


def test_plane_proof_requires_every_vertex_above_the_calibrated_table():
    scene=CollisionScene(vertical_scene());start=[[.10000012]]
    pairs,_=near_clearance_pairs(scene,start)
    recovery=ClearanceRecovery(scene,start,pairs,table_id='world')
    assert recovery._separating_table_pairs([[.099]],[[.12]])==[]
    recovery.table_id=None
    assert recovery._separating_table_pairs(start,[[.12]])==[]


def test_all_below_margin_pairs_are_discovered_not_only_the_closest():
    data=make();second=deepcopy(data['colliders'][1]);second['id']='second_world'
    second['vertices']=(np.array(second['vertices'])+[.000004,0.,0.]).tolist()
    data['colliders'].append(second);scene=CollisionScene(data)
    pairs,result=near_clearance_pairs(scene,[[-.100186]])
    assert len(pairs)==2 and result['clear']
    pairs,_=near_clearance_pairs(scene,[[-.100186]],allowed_pairs=[('finger','world')])
    assert pairs==[('finger','second_world')] or pairs==[('second_world','finger')]


def test_actual_overlap_is_rejected_even_within_nonworsening_noise_tolerance():
    scene=CollisionScene(make(-.100005));start=[[-.100005]]
    pairs,_=near_clearance_pairs(scene,start);recovery=ClearanceRecovery(scene,start,pairs)
    assert not recovery.check_actual([[-.099999]])
    assert recovery.failure_reason=='measured_positive_gap_not_preserved'
    assert not recovery.samples[-1]['accepted']
    assert near_clearance_pairs(scene,[[-.099999]])[0]==[]


def test_retreat_swept_check_still_rejects_a_different_obstacle():
    data=make();other=deepcopy(data['colliders'][1]);other['id']='other_wall'
    other['vertices']=(np.array(other['vertices'])+[-.3,0.,0.]).tolist()
    data['colliders'].append(other);scene=CollisionScene(data);start=[[-.100186]]
    pairs,_=near_clearance_pairs(scene,start);recovery=ClearanceRecovery(scene,start,pairs)
    result=recovery.check_segment(start,[[-.25]])
    # Large segment may also hit the discrete sample cap: neither can admit it.
    assert result['status']!='clear'
    result=recovery.check_segment(start,[[-.21]])
    assert result['reason']=='positive_gap_swept_nonintersection_not_certified'
    # The recovery pair may be below padding, but a NEW near pair may not.
    target=[[-.19985]]
    assert recovery.check_segment(start,target)['status']=='clear'
    assert scene.check_path(start,target,allowed_pairs=recovery.pairs)['status']!='clear'


def control():
    from sim.envs.robodojo.checked_control import CheckedController
    from sim.envs.robodojo.x5_calibration import link_to_tcp
    from sim.envs.robodojo.scene_geometry import forward_kinematics
    scene=CollisionScene(make());c=CheckedController.__new__(CheckedController)
    c.roles=['left'];c.allowed_pairs=[];c.contact_grants={};c.gripper_intents=[.5]
    c.world=SimpleNamespace(table_id="world",scene=scene,refresh=lambda:None,positions=lambda:[np.array([-.100186])],public_policy=lambda:{})
    c.native=SimpleNamespace(sim=SimpleNamespace(physics_dt=.004),robot_manager=SimpleNamespace(
        robot_list=[object()],scene=SimpleNamespace(env_origins=np.zeros((1,3)))))
    c.adapter=SimpleNamespace(_terminal=False,_last_obs={'proprio':{'arms':{'left':{'gripper_state':{'openness':.5}}}}})
    c.tcp=lambda i:link_to_tcp(forward_kinematics(scene.snapshot['robots'][i],[-.100186])['link6'])
    c._contact_scene=lambda s:s
    return c


def test_normal_move_routes_near_margin_to_bounded_recovery_before_ik():
    from sim.envs.robodojo.motion_planning import execute_request
    c=control();target=c.tcp(0);target[0,3]-=.01
    c._recovery_move=lambda *a,**kw:{'called_recovery':True}
    assert execute_request(c,{'left':[target]},timeout_s=2.,max_actions=None)=={'called_recovery':True}


def test_gripper_near_gap_mode_is_explicit_and_does_not_claim_existing_penetration():
    c=control();modes=[]
    c._action=lambda *a:modes.append(c.recovery.mode)
    r=c.gripper('left',.5)
    assert modes==['positive_clearance_retreat']*4
    assert r['host_adjustments'][0]['reason']=='positive_gap_below_normal_margin'
    assert r['recovery_diagnostics']['initial_minimum_gap_m']>0
    assert r['recovery_diagnostics']['actual_gap_must_remain_positive']
    boundary=r['initial_clearance_boundary']
    assert boundary['classification']=='positive_gap_not_overlap'
    assert boundary['contacts'][0]['measured_signed_clearance_m']>0
    assert boundary['contacts'][0]['participants'][0]['kind'] in ('robot','static_support')


def test_clearance_recovery_still_rejects_long_or_rotating_requests_before_actions():
    c=control();target=c.tcp(0);target[0,3]-=.1
    c._action=lambda *a:(_ for _ in ()).throw(AssertionError('must not move'))
    c._solve_pose=lambda *a:(_ for _ in ()).throw(AssertionError('must reject bounds first'))
    r=c._recovery_move({'left':[target]})
    assert r['native_actions']==r['physics_steps']==0
    assert r['stop']['reason']=='clearance_recovery_requires_short_translation'


def mixed_scene(*, opposed=False):
    data=make(-.0998)
    second=deepcopy(data['colliders'][1]);second['id']='near_wall'
    second['vertices']=(np.array(second['vertices'])+[-.1999 if opposed else .0003,0.,0.]).tolist()
    data['colliders'].append(second)
    return CollisionScene(data)


def mixed_recovery(scene, start):
    from sim.envs.robodojo.boundary_recovery import BoundaryRecovery
    from sim.envs.robodojo.clearance_recovery import include_near_clearance
    check=scene.check(start)
    existing=BoundaryRecovery(scene,start,check['contacts'])
    return existing,include_near_clearance(existing,scene,start)


def test_mixed_recovery_fixes_initial_padding_deadlock_without_ignoring_either_group():
    scene=mixed_scene();start=[[-.0998]];target=[[-.11]]
    old,recovery=mixed_recovery(scene,start)
    assert scene.check_path(start,target,allowed_pairs=old.pairs)['reason']=='insufficient_clearance'
    assert len(old.pairs)==1 and len(recovery.pairs)==2
    assert scene.check_path(start,target,allowed_pairs=recovery.pairs)['status']=='clear'
    assert recovery.check_segment(start,target)['status']=='clear'
    assert recovery.check_segment(start,[[-.0997]])['recovery_group']=='existing_contact'
    assert recovery.check_actual([[-.100]])
    assert not recovery.check_actual([[-.0999]])  # worsens the best measured overlap
    assert recovery.last_check['maximum_deepening_m']>1e-5
    assert recovery.diagnostics()['positive_gap']['actual_gap_must_remain_positive']


def test_mixed_recovery_rejects_shrinking_positive_gap_even_when_overlap_improves():
    scene=mixed_scene(opposed=True);start=[[-.0998]]
    _,recovery=mixed_recovery(scene,start)
    assert recovery.check_segment(start,[[-.09985]])['recovery_group']=='initially_positive_gap'
    assert not recovery.check_actual([[-.099905]])
    assert recovery.failure_reason=='measured_positive_gap_not_preserved'


def test_mixed_recovery_keeps_new_obstacles_and_fresh_prediction_scene():
    scene=mixed_scene();start=[[-.0998]];_,recovery=mixed_recovery(scene,start)
    data=deepcopy(scene.snapshot);other=deepcopy(data['colliders'][1]);other['id']='new_wall'
    other['vertices']=(np.array(other['vertices'])+[-.21,0.,0.]).tolist()
    data['colliders'].append(other);updated=CollisionScene(data)
    recovery.scene=updated
    assert recovery.existing.scene is recovery.positive.scene is updated
    result=recovery.check_segment(start,[[-.115]])
    assert result['status']!='clear'
    assert result['reason']=='positive_gap_swept_nonintersection_not_certified'
    # A new near pair, even without overlap, still fails the normal guard.
    assert updated.check_path(start,[[-.10985]],allowed_pairs=recovery.pairs)['status']!='clear'


def test_no_extra_near_pairs_keeps_existing_recovery_unchanged():
    from sim.envs.robodojo.clearance_recovery import include_near_clearance
    from sim.envs.robodojo.boundary_recovery import BoundaryRecovery
    scene=CollisionScene(make(-.0998));start=[[-.0998]]
    old=BoundaryRecovery(scene,start,scene.check(start)['contacts'])
    assert include_near_clearance(old,scene,start) is old


def test_controller_installs_both_recovery_groups_before_first_native_action():
    c=control();scene=mixed_scene();start=[np.array([-.0998])]
    c.world.scene=scene;c.world.positions=lambda:start
    target=c.tcp(0);target[0,3]-=.01
    c._solve_pose=lambda *args:(np.array([-.11]),None)
    c._grippers=lambda:[]
    modes=[]
    def action(*args):
        modes.append(c.recovery.mode)
        c.stopped={'reason':'test_stopped_before_physics'}
    c._action=action
    receipt=c._recovery_move({'left':[target]})
    assert modes==['mixed_contact_and_positive_clearance_retreat']
    assert receipt['initial_collision'] is not None
    assert receipt['initial_clearance_boundary']['classification']=='positive_gap_not_overlap'
    assert receipt['recovery_diagnostics']['existing_contact_mode']=='sampled_nonworsening_retreat'
