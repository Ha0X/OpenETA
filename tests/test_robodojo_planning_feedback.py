import json
from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.checked_control import CheckedController
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.ik_candidates import validated_candidates
from sim.envs.robodojo.scene_geometry import forward_kinematics
from sim.envs.robodojo.x5_calibration import link_to_tcp, tcp_to_link
from test_robodojo_collision_geometry import fixture


def data():
    value = fixture(start=-1., wall_center=10.)
    r = value['robots'][0]; r['role'] = 'left'
    r['body_frames']['link6'] = r['body_frames'].pop('tool')
    r['joints'][0]['child'] = 'link6'
    value['colliders'][0]['body'] = 'link6'
    return value


def test_physical_fk_limits_and_nonfinite_candidates_are_separate_failures():
    r = data()['robots'][0]
    target = link_to_tcp(forward_kinematics(r, [-1.])['link6'])
    valid, rejected = validated_candidates(r, [-1.], [[-1.], [-.5], [3.], [float('nan')]], target, np.zeros(3))
    assert len(valid) == 1
    assert rejected == {'nonfinite_or_shape': 1, 'joint_limit': 1, 'physical_fk': 1}


def test_later_waypoint_failure_prevents_all_physics_and_reports_index():
    c = CheckedController.__new__(CheckedController)
    scene = CollisionScene(data())
    c.world = SimpleNamespace(scene=scene, refresh=lambda: None, positions=lambda: [np.array([-1.])], public_policy=lambda: {})
    c.roles = ['left']; c.contact_grants = {}; c.allowed_pairs = []
    c.native = SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1, 3)))))
    c.tcp = lambda i: link_to_tcp(forward_kinematics(scene.snapshot['robots'][0], [-1.])['link6'])
    c.adapter = SimpleNamespace(_terminal=False)
    def solve(i, pose, current):
        x = tcp_to_link(pose)[0, 3]
        return ([np.array([x])] if x < -.92 else []), {'valid_unique_candidates': int(x < -.92)}
    c._pose_candidates = solve
    def action(*args):
        raise AssertionError('No physics before all requested waypoints pass')
    c._action = action
    a = c.tcp(0); a[0, 3] += .04
    b = c.tcp(0); b[0, 3] += .12
    result = c.move({'left': [a, b]})
    assert result['native_actions'] == result['physics_steps'] == 0
    assert result['stop']['endpoint'] == 1
    assert result['stop']['status'] == 'unknown'
    assert result['stop']['reason'] == 'ik_search_no_valid_solution'


def test_nested_planner_collision_feedback_hides_asset_paths():
    c = CheckedController.__new__(CheckedController)
    c.roles = ['left']
    c.world = SimpleNamespace(scene=CollisionScene(data()), table_id='private_table')
    check = {'detail': {'contacts': [{'a': 'world', 'b': 'finger', 'robot_a': None, 'robot_b': 0}],
                        'clearance_m': 0., 'closest_pair': ['world', 'finger']}}
    public = c._public_plan({'joint_planner': {'last_check': check}})
    assert all(raw not in json.dumps(public) for raw in ('"world"','"finger"','private_table'))
    detail = public['joint_planner']['last_check']
    assert detail['contacts'][0]['participants'][1] == {'kind': 'robot', 'role': 'left', 'part': 'link6'}
    assert detail['closest_pair']['geometry_a']=='geometry_1'
    assert detail['closest_pair']['participants'][1]=={'kind':'robot','role':'left','part':'link6'}


def test_failed_ik_residuals_use_physical_fk_and_exclude_invalid_limits():
    from scipy.spatial.transform import Rotation
    from sim.envs.robodojo.ik_candidates import sampled_seed_residuals
    schema = data()['robots'][0]
    target = link_to_tcp(forward_kinematics(schema, [-.6])['link6'])
    target[:3, :3] = Rotation.from_euler('z', .8).as_matrix() @ target[:3, :3]
    origin = np.array([.1, -.2, .3])
    target[:3, 3] -= origin
    candidates = [[-.8], [-.5], [3.], [float('nan')]]
    before = target.copy()
    result = sampled_seed_residuals(schema, [-1.], candidates, target, origin)
    assert result['physical_limit_valid_samples'] == 2
    best = result['best_balanced_sample']
    assert abs(best['position_error_m']-.1)<1e-9
    assert abs(best['orientation_error_rad']-.8)<1e-9
    assert not result['collision_checked'] and not result['target_changed']
    assert np.array_equal(before, target)
    assert all('joint' not in key and 'pose' not in key for key in best)


def test_failed_ik_without_physical_samples_does_not_invent_error_or_goal():
    from sim.envs.robodojo.ik_candidates import sampled_seed_residuals
    schema = data()['robots'][0]
    target = link_to_tcp(forward_kinematics(schema, [-1.])['link6'])
    result = sampled_seed_residuals(schema, [-1.], [[3.], [float('nan')], []], target, np.zeros(3))
    assert result['physical_limit_valid_samples'] == 0
    assert 'best_balanced_sample' not in result


def test_failed_solver_flag_does_not_discard_a_physically_valid_candidate():
    from sim.envs.robodojo.ik_candidates import ordered_seed_candidates
    schema = data()['robots'][0]
    target = link_to_tcp(forward_kinematics(schema, [-.5])['link6'])
    js = SimpleNamespace(joint_names=['slide'], position=np.array([[[-.5], [3.], [float('nan')], [-1.]]]))
    candidates = ordered_seed_candidates(js, [False, True, True, True], ['slide'])
    valid, rejected = validated_candidates(schema, [-1.], candidates, target, np.zeros(3))
    assert len(valid) == 1 and np.allclose(valid[0], [-.5])
    assert rejected == {'nonfinite_or_shape': 1, 'joint_limit': 1, 'physical_fk': 1}


def test_all_seed_mapping_preserves_named_joint_order_and_rejects_bad_identity():
    import pytest
    from sim.envs.robodojo.ik_candidates import ordered_seed_candidates
    js = SimpleNamespace(joint_names=['b', 'a'], position=np.array([[[2., 1.], [4., 3.]]]))
    assert np.array_equal(ordered_seed_candidates(js, [False, True], ['a','b']), [[1.,2.],[3.,4.]])
    with pytest.raises(ValueError, match='seed shape'):
        ordered_seed_candidates(js, [False], ['a','b'])
    js.joint_names=['a','a']
    with pytest.raises(ValueError, match='identity'):
        ordered_seed_candidates(js, [False, True], ['a','b'])


def test_physically_valid_false_flag_candidate_still_requires_collision_admission():
    import time
    from sim.envs.robodojo.ik_candidates import ordered_seed_candidates
    from sim.envs.robodojo.motion_planning import combined_goals
    value=data()
    value['colliders'][1]['vertices'] = (np.array(value['colliders'][1]['vertices']) - [10.5,0.,0.]).tolist()
    scene=CollisionScene(value);schema=value['robots'][0]
    target=link_to_tcp(forward_kinematics(schema,[-.5])['link6'])
    candidates=ordered_seed_candidates(SimpleNamespace(joint_names=['slide'],position=[[-.5]]),[False],['slide'])
    c=CheckedController.__new__(CheckedController)
    c.roles=['left'];c.allowed_pairs=[];c.world=SimpleNamespace(scene=scene)
    c.native=SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1,3)))))
    c.candidate_solver=SimpleNamespace(solve=lambda *a:(candidates,{'converged_seeds':0}))
    goals,detail=combined_goals(c,[target],[[-1.]],scene,[0],time.monotonic()+5)
    assert not goals and detail['reason']=='ik_candidates_in_collision'
    assert detail['ik'][0]['valid_unique_candidates']==1
    assert detail['collision_rejected_candidates']==1


def test_budget_feedback_names_limiting_parts_without_exposing_asset_ids():
    c=CheckedController.__new__(CheckedController);c.roles=['left']
    value=data();value['colliders'][1]['vertices']=(np.asarray(value['colliders'][1]['vertices'])-[10.,0.,0.]).tolist()
    c.world=SimpleNamespace(scene=CollisionScene(value),table_id='private_table')
    raw=c.world.scene.check_path([[-1.]],[[1.]],max_nodes=1)
    public=c._public_check(raw)
    assert public['status']=='unknown' and not public['contacts']
    assert public['clearance_limiting_pair_count']==1
    assert {'kind':'robot','role':'left','part':'link6'} in public['clearance_limiting_pairs'][0]['participants']
    assert '"finger"' not in json.dumps(public) and '"world"' not in json.dumps(public)


def test_cartesian_path_collision_returns_evidence_without_joint_detour(monkeypatch):
    from sim.envs.robodojo.motion_planning import plan_request
    from sim.envs.robodojo.joint_planning import JointPlanner
    def forbidden(*args,**kwargs):
        raise AssertionError('No automatic joint-space detours')
    monkeypatch.setattr(JointPlanner,'plan',forbidden)
    value=data()
    # Restore the obstacle between two free endpoints.
    for vertex in value['colliders'][1]['vertices']:vertex[0]-=10.
    scene=CollisionScene(value)
    c=CheckedController.__new__(CheckedController)
    c.world=SimpleNamespace(scene=scene,refresh=lambda:None,positions=lambda:[np.array([-1.])])
    c.roles=['left'];c.contact_grants={};c.allowed_pairs=[]
    c.native=SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1,3)))))
    c.tcp=lambda i:link_to_tcp(forward_kinematics(scene.snapshot['robots'][0],[-1.])['link6'])
    c._pose_candidates=lambda i,pose,current:([np.array([tcp_to_link(pose)[0,3]])],{'count':1})
    target=c.tcp(0);target[0,3]+=2.
    trajectory,detail=plan_request(c,{'left':[target]},5.)
    assert trajectory is None
    assert detail['reason']=='requested_cartesian_path_not_certified'
    assert detail['phase']=='path_preview' and detail['physics_advanced'] is False
    assert detail['automatic_spatial_replanning'] is False
    assert detail['cartesian_check']


def test_clear_cartesian_edges_get_mesh_budget_but_keep_request_deadline(monkeypatch):
    from sim.envs.robodojo.motion_planning import plan_request
    scene = CollisionScene(data())
    c = CheckedController.__new__(CheckedController)
    c.world = SimpleNamespace(scene=scene, refresh=lambda: None, positions=lambda: [np.array([-1.])])
    c.roles = ['left']; c.contact_grants = {}; c.allowed_pairs = []
    c.native = SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=np.zeros((1, 3)))))
    c.tcp = lambda i: link_to_tcp(forward_kinematics(scene.snapshot['robots'][0], [-1.])['link6'])
    c._pose_candidates = lambda i, pose, current: ([np.array([tcp_to_link(pose)[0, 3]])], {'count': 1})
    budgets = []
    original = scene.check_path
    def checked(*args, **kwargs):
        budgets.append(kwargs['timeout_s'])
        assert kwargs['max_nodes'] == 512
        return original(*args, **kwargs)
    monkeypatch.setattr(scene, 'check_path', checked)
    target = c.tcp(0); target[0, 3] += .04
    for request_budget in (10., .5):
        budgets.clear()
        path, feedback = plan_request(c, {'left': [target]}, request_budget)
        assert path is not None and budgets
        assert max(budgets) <= min(4., request_budget)
        if request_budget == 10.:
            assert min(budgets) == 4.
        assert feedback['cartesian_edge_check_time_limit_s'] == 4.


def representative_fixture(clear_second=False):
    schema={'joint_names':['j'],'arm_joint_names':['j']}
    def check(q,**kwargs):
        value=float(q[0][0]);return {'clear':clear_second and value==1.,'tag':value}
    scene=SimpleNamespace(snapshot={'robots':[schema]},check=check)
    c=SimpleNamespace(roles=['left'],allowed_pairs=[],
        _pose_candidates=lambda *args:([np.array([3.]),np.array([.1]),np.array([1.])],{}))
    return c,scene


def test_failed_endpoint_reports_nearest_checked_candidate_not_last_remote_posture():
    import time
    from sim.envs.robodojo.motion_planning import combined_goals
    c,scene=representative_fixture()
    goals,r=combined_goals(c,[None],[[0.]],scene,[0],time.monotonic()+10.)
    assert not goals and r['candidate_combinations_tested']==3 and r['collision_rejected_candidates']==3
    assert r['detail']['tag']==.1 and r['collision_evidence']['candidate_rank']==1
    assert r['collision_evidence']['scope']=='one_rejected_ik_configuration_not_all_candidates'


def test_feasible_candidate_search_and_acceptance_are_unchanged():
    import time
    from sim.envs.robodojo.motion_planning import combined_goals
    c,scene=representative_fixture(clear_second=True)
    goals,r=combined_goals(c,[None],[[0.]],scene,[0],time.monotonic()+10.)
    assert len(goals)==1 and goals[0][0][0]==1. and r['candidate_combinations_tested']==3
    assert r['detail'] is None and r['collision_evidence'] is None


def test_timeout_before_checking_candidates_has_no_invented_collision_evidence(monkeypatch):
    import sim.envs.robodojo.motion_planning as module
    moments=iter([0.,2.]);monkeypatch.setattr(module.time,'monotonic',lambda:next(moments))
    c,scene=representative_fixture()
    goals,r=module.combined_goals(c,[None],[[0.]],scene,[0],1.)
    assert not goals and r['reason']=='ik_preview_budget' and r['candidate_combinations_tested']==0
    assert r['detail'] is None and r['collision_evidence'] is None
