from pathlib import Path
import importlib.util
import time
from types import SimpleNamespace
import numpy as np


def staged():
    from sim.envs.robodojo import motion_planning
    return motion_planning


def fixture():
    control=SimpleNamespace(roles=['left'],allowed_pairs=[],
        _pose_candidates=lambda *a:([np.array([float(i)]) for i in range(11)],{}))
    checked=[]
    def check(q,**kwargs):
        checked.append(float(q[0][0]))
        return {'clear':q[0][0]==10.,'contacts':[],'clearance_m':0.}
    scene=SimpleNamespace(snapshot={'robots':[{'joint_names':['q'],'arm_joint_names':['q']}]},check=check)
    return control,scene,checked


def test_single_arm_checks_generated_candidates_beyond_first_eight():
    m=staged();c,s,checked=fixture()
    goals,detail=m.combined_goals(c,[np.eye(4)],[[0.]],s,[0],time.monotonic()+5)
    assert len(goals)==1 and goals[0][0][0]==10.
    assert checked==list(range(11))
    assert detail['candidate_combinations_tested']==11
    assert detail['candidate_search_truncated'] is False


def test_deadline_exhaustion_is_not_reported_as_candidate_collision(monkeypatch):
    m=staged();c,s,checked=fixture();clock=iter([0.,2.])
    monkeypatch.setattr(m,'time',SimpleNamespace(monotonic=lambda:next(clock)))
    goals,detail=m.combined_goals(c,[np.eye(4)],[[0.]],s,[0],1.)
    assert not goals and not checked
    assert detail['reason']=='ik_preview_budget'
    assert detail['candidate_search_truncated'] is True


def test_dual_candidate_search_remains_bounded_to_64_combinations():
    m=staged();c,s,checked=fixture();c.roles=['left','right']
    s.snapshot['robots']*=2
    s.check=lambda *a,**k:{'clear':False,'contacts':[],'clearance_m':0.}
    goals,detail=m.combined_goals(c,[np.eye(4),np.eye(4)],[[0.],[0.]],s,[0,1],time.monotonic()+5)
    assert not goals and detail['candidate_combinations_generated']==121
    assert detail['candidate_combinations_tested']==64
    assert detail['candidate_search_truncated'] is True


def test_cartesian_budget_exhaustion_returns_to_agent_without_detour(monkeypatch):
    from sim.envs.robodojo import checked_control,attached_geometry
    m=staged();clock=[0.];calls=[]
    monkeypatch.setattr(m,'time',SimpleNamespace(monotonic=lambda:clock[0]))
    scene=SimpleNamespace(snapshot={'robots':[{}]},check=lambda *a,**k:{'clear':True,'clearance_m':1.},
        check_path=lambda *a,**k:{'status':'clear'})
    c=SimpleNamespace(world=SimpleNamespace(scene=scene,refresh=lambda:None,positions=lambda:[[0.]]),
        native=SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=[[0,0,0]]))),
        roles=['left'],contact_grants={},allowed_pairs=[],_contact_scene=lambda s:s,tcp=lambda i:np.eye(4))
    monkeypatch.setattr(attached_geometry,'attached_scene',lambda *a:(scene,[]))
    monkeypatch.setattr(checked_control,'cartesian_knots',lambda a,b:[b]*10)
    def goals(*a):
        calls.append(1);clock[0]+=1 if len(calls)==1 else 10
        return [[[0.]]],{}
    monkeypatch.setattr(m,'combined_goals',goals)
    trajectory,detail=m.plan_request(c,{'left':[np.eye(4)]},45.)
    assert trajectory is None and len(calls)==6
    assert detail['reason']=='requested_cartesian_path_not_certified'
    assert detail['status']=='unknown'
    assert detail['cartesian_check']['reason']=='cartesian_preview_budget_exhausted'
    assert detail['automatic_spatial_replanning'] is False
    assert detail['physics_advanced'] is False


def test_independent_candidate_track_checks_beyond_first_eight(monkeypatch):
    from sim.envs.robodojo import coordinated_motion,checked_control
    c,s,checked=fixture();c.tcp=lambda i:np.eye(4);s.items=[]
    def check_path(start,trial,**kw):
        checked.append(float(trial[0][0]))
        return {'status':'clear' if trial[0][0]==10 else 'blocked'}
    s.check_path=check_path
    monkeypatch.setattr(checked_control,'cartesian_knots',lambda a,b:[b])
    tracks,failure=coordinated_motion.candidate_tracks(c,{'left':[np.eye(4)]},s,[[0.]],.04,time.monotonic()+5)
    assert failure is None and tracks
    assert checked==list(range(11))


def test_cartesian_planner_requests_only_target_tcps_but_checks_all_articulations(monkeypatch):
    from sim.envs.robodojo import attached_geometry
    m=staged();seen=[]
    schemas=[{'joint_names':['q'],'arm_joint_names':['q']} for _ in range(3)]
    def check(q,**kwargs):
        assert len(q)==3 and q[2][0]==.7
        seen.append('geometry')
        return {'clear':True,'clearance_m':1.}
    def path(a,b,**kwargs):
        check(a);check(b);return {'status':'clear'}
    scene=SimpleNamespace(snapshot={'robots':schemas},check=check,check_path=path)
    def tcp(index):
        assert index<2, 'Auxiliary robot is not a controllable TCP'
        seen.append(('tcp',index));return np.eye(4)
    def candidates(index,pose,q):
        assert index==0
        return [np.array([pose[0,3]])],{}
    c=SimpleNamespace(world=SimpleNamespace(scene=scene,refresh=lambda:None,positions=lambda:[[0.],[0.],[.7]]),
        roles=['left','right'],contact_grants={},allowed_pairs=[],_contact_scene=lambda s:s,tcp=tcp,
        _pose_candidates=candidates)
    monkeypatch.setattr(attached_geometry,'attached_scene',lambda *a:(scene,[]))
    target=np.eye(4);target[0,3]=.01
    trajectory,detail=m.plan_request(c,{'left':[target]},5.)
    assert trajectory and detail['status']=='clear'
    assert ('tcp',0) in seen and ('tcp',1) in seen and 'geometry' in seen
    assert trajectory[-1][0][0]==.01 and trajectory[-1][2][0]==.7
