import copy

import numpy as np

from sim.envs.robodojo.collision_bounds import RelativeMotionBounds
from sim.envs.robodojo.scene_geometry import forward_kinematics


def scene():
    def joint(name,parent,child,kind,offset):
        t=np.eye(4);t[0,3]=offset
        return {'name':name,'parent':parent,'child':child,'kind':kind,'axis':'Z',
                'parent_frame':t.tolist(),'child_frame':np.eye(4).tolist()}
    robot={'base_link':'base','joint_names':['root','a','b'],'joint_limits':[[-2,2]]*3,
        'body_frames':{name:np.eye(4).tolist() for name in ['base','common','left','right']},
        'joints':[joint('root','base','common','revolute',.1),
                  joint('a','common','left','revolute',.3),joint('b','common','right','revolute',-.3)]}
    return {'robots':[robot],'colliders':[{'robot_index':0,'body':body,'vertices':[[.2,0,0],[0,.1,0]]}
                                         for body in ['left','right']]}


def test_common_upstream_rotation_does_not_expand_relative_clearance_bound():
    s=scene();bound=RelativeMotionBounds(s).for_segment([[0,0,0]],[[1.5,0,0]],[(0,1)])
    assert bound[(0,1)]==0


def test_separation_change_always_bounded_for_branch_joint_motion():
    s=scene();b=RelativeMotionBounds(s);rng=np.random.default_rng(54)
    for _ in range(100):
        start=rng.uniform(-1,1,3);target=rng.uniform(-1,1,3)
        fk0=forward_kinematics(s['robots'][0],start);fk1=forward_kinematics(s['robots'][0],target)
        def distances(fk):
            points=[]
            for item in s['colliders']:
                t=fk[item['body']];points.append(np.array(item['vertices'])@t[:3,:3].T+t[:3,3])
            return np.linalg.norm(points[0][:,None,:]-points[1][None,:,:],axis=-1)
        assert np.max(abs(distances(fk1)-distances(fk0))) <= b.for_segment([start],[target],[(0,1)])[(0,1)]+1e-10


def test_interarm_pair_keeps_each_articulations_upstream_motion():
    s=scene();s['robots'].append(copy.deepcopy(s['robots'][0]));s['colliders'][1]['robot_index']=1
    bound=RelativeMotionBounds(s).for_segment([[0,0,0],[0,0,0]],[[1,0,0],[1,0,0]],[(0,1)])
    assert bound[(0,1)]>0


def test_coefficient_cache_keeps_fresh_deltas_and_invalidates_pair_directory():
    s=scene();s['colliders'].append({'robot_index':None,'body':None,'vertices':[[0,0,0]]})
    b=RelativeMotionBounds(s);pairs=[(0,1)]
    first=b.for_segment([[0,0,0]],[[0,.1,.2]],pairs)[(0,1)]
    matrix=b._coefficients
    second=b.for_segment([[0,0,0]],[[0,.2,.4]],pairs)[(0,1)]
    assert b._coefficients is matrix and np.isclose(second,2*first)
    pairs.append((0,2))
    result=b.for_segment([[0,0,0]],[[1,0,0]],pairs)
    assert b._coefficients is not matrix
    assert result[(0,1)]==0 and result[(0,2)]>0


def test_cached_bounds_match_original_chain_sum_with_prismatic_and_fixed_joints():
    s=scene();robot=s['robots'][0]
    robot['joints'][1]['kind']='prismatic';robot['joints'][2]['kind']='fixed'
    s['robots'].append(copy.deepcopy(robot));s['colliders'][1]['robot_index']=1
    s['colliders'].append({'robot_index':None,'body':None,'vertices':[[0,0,0]]})
    b=RelativeMotionBounds(s);rng=np.random.default_rng(23);pairs=[(0,1),(0,2),(1,2)]
    for _ in range(25):
        start=rng.uniform(-2,2,size=(2,3));target=rng.uniform(-2,2,size=(2,3))
        values=b.for_segment(start,target,pairs)
        for i,j in pairs:
            expected=0.
            for k in (i,j):
                index=s['colliders'][k]['robot_index']
                if index is not None:
                    for name,kind,radius in b.chains[k]:
                        if kind!='fixed':
                            slot=robot['joint_names'].index(name)
                            expected+=abs(target[index,slot]-start[index,slot])*(radius if kind=='revolute' else 1.)
            assert np.isclose(values[(i,j)],expected,rtol=1e-14,atol=1e-15)


def test_prismatic_overtravel_expands_swept_bound_without_changing_joint_limits():
    s=scene();r=s['robots'][0];r['joints'][1].update(kind='prismatic',axis='X')
    s['colliders'].append({'robot_index':None,'body':None,'vertices':[[0,0,0]]})
    b=RelativeMotionBounds(s);start=[[0.,100.,0.]];target=[[1.,100.,0.]]
    bound=b.for_segment(start,target,[(0,2)])[(0,2)]
    a=forward_kinematics(r,start[0])['left'];z=forward_kinematics(r,target[0])['left']
    point=np.array([.2,0,0,1.])
    assert np.linalg.norm((z@point-a@point)[:3]) <= bound
    assert r['joint_limits'][1]==[-2,2]
    # A stationary segment still has zero motion, even outside the DOF limits.
    assert b.for_segment(start,start,[(0,2)])[(0,2)]==0
