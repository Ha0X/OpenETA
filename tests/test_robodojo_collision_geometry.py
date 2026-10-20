from itertools import product

import numpy as np
import pytest

from sim.envs.robodojo.scene_geometry import forward_kinematics, collision_envelope
from sim.envs.robodojo.collision_geometry import CollisionScene


def fixture(start=-1., wall_center=0., wall_radius=.05):
    identity=np.eye(4);tool=identity.copy();tool[0,3]=start
    cube=np.array(list(product([-.05,.05],repeat=3)))
    world=np.array(list(product([-wall_radius,wall_radius],repeat=3)));world[:,0]+=wall_center
    def shape(name,vertices,robot,body):
        return {'id':name,'collider_path':name,'robot_index':robot,'body':body,
            'vertices':vertices.tolist(),'triangles':[],'approximation':'convexHull'}
    return {'robots':[{'base_link':'base','joint_names':['slide'],'arm_joint_names':['slide'],
        'joint_positions':[start],'joint_limits':[[-2.,2.]],'body_frames':{'base':identity.tolist(),'tool':tool.tolist()},
        'joints':[{'name':'slide','parent':'base','child':'tool','kind':'prismatic','axis':'X',
                   'parent_frame':identity.tolist(),'child_frame':identity.tolist(),'collision_enabled':False}]}],
        'colliders':[shape('finger',cube,0,'tool'),shape('world',world,None,None)],
        'filtered_pairs':[],'unsupported':[]}


def test_path_checks_obstacle_between_clear_endpoints():
    scene=CollisionScene(fixture())
    assert scene.check([[-1.]])['clear'] and scene.check([[1.]])['clear']
    result=scene.check_path([[-1.]],[[1.]])
    assert result['status']=='blocked'
    assert result['reason'] in {'geometry_collision','insufficient_clearance'}
    assert result['samples']>1


def test_free_path_uses_swept_bound_and_does_not_mutate_source():
    data=fixture();scene=CollisionScene(data)
    assert scene.check_path([[-1.]],[[-.7]])['status']=='clear'
    assert data['robots'][0]['joint_positions']==[-1.]


def test_solid_containment_is_collision_even_without_surface_crossing():
    assert not CollisionScene(fixture(start=0,wall_radius=.5)).check([[0.]])['clear']


def test_missing_geometry_is_unknown_and_bad_fk_is_rejected():
    data=fixture();data['unsupported']=[{'reason':'unsupported_shape'}]
    with pytest.raises(ValueError,match='coverage'):CollisionScene(data)
    data=fixture();data['robots'][0]['body_frames']['tool'][0][3]=.4
    with pytest.raises(ValueError,match='FK disagreement'):CollisionScene(data)


def test_certificate_exhaustion_does_not_claim_clear():
    scene=CollisionScene(fixture())
    assert scene.check_path([[-1.]],[[1.]],max_nodes=0)['status']=='unknown'
    assert scene.check_path([[-1.]],[[3.]])['reason']=='joint_limit'


def test_physical_joint_frames_include_child_offset_and_rotation_axis():
    parent=np.eye(4);parent[0,3]=1
    child=np.eye(4);child[0,3]=.2
    robot={'base_link':'base','body_frames':{'base':np.eye(4).tolist(),'tip':np.eye(4).tolist()},
        'joint_names':['hinge'],'joints':[{'name':'hinge','parent':'base','child':'tip','kind':'revolute',
        'axis':'Z','parent_frame':parent.tolist(),'child_frame':child.tolist()}]}
    result=forward_kinematics(robot,[np.pi/2])['tip']
    assert np.allclose(result[:3,3],[1.,-.2,0.])
    assert np.allclose(result[:3,0],[0.,1.,0.])


def test_native_bounding_box_is_built_before_rotation():
    tetra=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    vertices,faces,kind=collision_envelope(tetra,[],'boundingCube')
    assert kind=='convexHull' and len(vertices)==8
    assert any(np.allclose(v,[1.,1.,1.]) for v in vertices)
    assert len(faces)==12


def test_batched_broad_phase_matches_exhaustive_fcl_with_limits_and_extra_pairs():
    import copy
    data=fixture()
    for index in range(12):
        item=copy.deepcopy(data['colliders'][1])
        item['id']=item['collider_path']=f'world_{index}'
        item['vertices']=(np.asarray(item['vertices'])+[index*.17-.8, .13*(index%3), 0]).tolist()
        data['colliders'].append(item)
    scene=CollisionScene(data);rng=np.random.default_rng(81)
    extra=[(1,2),(1,2)]  # World/world pairs are explicitly reintroduced once.
    pairs=list(dict.fromkeys([*scene.pairs,*extra]))
    allowed=[('finger','world_3')]
    for x in rng.uniform(-1.5,1.5,30):
        limits={pair:float(rng.uniform(.0002,.2)) for pair in pairs}
        value=scene.check([[x]],allowed_pairs=allowed,clearance_limit=1.,clearance_limits=limits,extra_pairs=extra)
        contacts=[];uncertified=[];minimum=1.
        for i,j in pairs:
            if frozenset((scene.items[i]['id'],scene.items[j]['id'])) in map(frozenset,allowed):continue
            hit=scene.fcl.CollisionResult()
            scene.fcl.collide(scene.objects[i],scene.objects[j],scene.fcl.CollisionRequest(),hit)
            distance=0. if hit.is_collision else scene.fcl.distance(scene.objects[i],scene.objects[j],
                scene.fcl.DistanceRequest(),scene.fcl.DistanceResult())
            minimum=min(minimum,distance)
            if hit.is_collision:contacts.append((scene.items[i]['id'],scene.items[j]['id']))
            if distance<=limits[(i,j)]:uncertified.append((i,j))
        assert [(c['a'],c['b']) for c in value['contacts']]==contacts
        assert value['uncertified_pairs']==uncertified
        assert value['clear']==(not contacts)
        assert 0<=value['clearance_m']<=minimum+1e-12


def test_broad_phase_handles_empty_pair_directory_and_touching_threshold():
    scene=CollisionScene(fixture(start=-.3))
    pair=scene.pairs[0]
    gap=scene.check([[-.3]])['clearance_m']
    assert scene.check([[-.3]],clearance_limits={pair:gap})['uncertified_pairs']==[pair]
    scene.pairs=[]
    assert scene.check([[-.3]],clearance_limit=.1)['clearance_m']==.1


@pytest.mark.parametrize('budget_kind',['nodes','time'])
def test_budget_feedback_retains_last_uncertified_pair_without_claiming_collision(monkeypatch,budget_kind):
    import sim.envs.robodojo.collision_geometry as geometry
    scene=CollisionScene(fixture())
    if budget_kind=='time':
        clock=iter([0.,0.,2.])
        monkeypatch.setattr(geometry.time,'monotonic',lambda:next(clock))
    result=scene.check_path([[-1.]],[[1.]],max_nodes=1 if budget_kind=='nodes' else 2048,timeout_s=1.)
    assert result['status']=='unknown' and result['samples']==1
    assert result['reason']=='clearance_certificate_budget'
    assert result['detail']['contacts']==[]
    assert result['detail']['uncertified_pairs']
    assert result['uncertified_segment_fraction']==[0.,1.]
    assert 'not_a_measured_collision' in result['evidence']
