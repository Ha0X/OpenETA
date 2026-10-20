import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from sim.envs.robodojo.motion_corridor import OrientationArc


def pose(rotation):
    t=np.eye(4);t[:3,:3]=rotation.as_matrix();return t


def test_large_commanded_turn_includes_its_middle():
    a=pose(Rotation.identity());b=pose(Rotation.from_euler('z',180,degrees=True))
    arc=OrientationArc(a,b)
    for degrees in (0,30,60,90,120,150,180):
        assert arc.distance(pose(Rotation.from_euler('z',degrees,degrees=True)))<1e-6
    assert arc.distance(pose(Rotation.from_euler('x',90,degrees=True)))>.35


def test_zero_rotation_still_enforces_original_orientation_limit():
    a=pose(Rotation.identity());arc=OrientationArc(a,a)
    assert abs(arc.distance(pose(Rotation.from_euler('x',.2)))-.2)<1e-7
    assert arc.distance(pose(Rotation.from_euler('x',.4)))>.35


def test_arc_does_not_allow_rotation_past_requested_endpoints():
    a=pose(Rotation.identity());b=pose(Rotation.from_euler('z',60,degrees=True))
    arc=OrientationArc(a,b)
    assert abs(arc.distance(pose(Rotation.from_euler('z',-30,degrees=True)))-np.pi/6)<1e-7
    assert abs(arc.distance(pose(Rotation.from_euler('z',100,degrees=True)))-np.deg2rad(40))<1e-7


def test_closed_form_matches_dense_geodesic_search_for_arbitrary_rotations():
    rng=np.random.default_rng(7)
    for _ in range(15):
        endpoints=Rotation.random(2,random_state=rng);query=Rotation.random(random_state=rng)
        arc=OrientationArc(pose(endpoints[0]),pose(endpoints[1]))
        samples=Slerp([0,1],endpoints)(np.linspace(0,1,10001))
        reference=np.min((samples.inv()*query).magnitude())
        assert abs(arc.distance(pose(query))-reference)<.0004


def test_large_turn_with_uncertified_path_does_not_trigger_joint_detour(monkeypatch):
    from types import SimpleNamespace
    from sim.envs.robodojo import motion_planning as module
    from sim.envs.robodojo import attached_geometry
    scene=SimpleNamespace(snapshot={'robots':[{}]},check=lambda *a,**k:{'clear':True,'clearance_m':1.})
    control=SimpleNamespace(world=SimpleNamespace(scene=scene,refresh=lambda:None,positions=lambda:[[0.]]),
        native=SimpleNamespace(robot_manager=SimpleNamespace(scene=SimpleNamespace(env_origins=[[0,0,0]]))),
        roles=['left'],contact_grants={},allowed_pairs=[],_contact_scene=lambda s:s,tcp=lambda i:np.eye(4))
    monkeypatch.setattr(attached_geometry,'attached_scene',lambda *a:(scene,[]))
    calls=[]
    def goals(*a):
        calls.append(1)
        return ([[[np.pi]]],{}) if len(calls)==1 else ([],{'reason':'force_joint_fallback'})
    monkeypatch.setattr(module,'combined_goals',goals)
    from sim.envs.robodojo import checked_control
    monkeypatch.setattr(checked_control,'cartesian_knots',lambda a,b:[b])
    path,result=module.plan_request(control,{'left':[pose(Rotation.from_euler('z',np.pi))]},5.)
    assert path is None and len(calls)==2
    assert result['reason']=='requested_cartesian_path_not_certified'
    assert result['cartesian_check']['reason']=='force_joint_fallback'
    assert result['automatic_spatial_replanning'] is False
    assert result['physics_advanced'] is False
