from copy import deepcopy
import numpy as np
import pytest
from sim.envs.robodojo.collision_state import auxiliary_finger_positions
from sim.envs.robodojo.scene_geometry import forward_kinematics
from test_robodojo_attached_geometry import grasp_scene


def schema():
    world,_=grasp_scene();r=world.scene.snapshot['robots'][0]
    r.update(control_owner='support',finger_links=['link7','link8'])
    return r


def test_collision_state_matches_actual_finger_overtravel_without_changing_readback():
    r=schema();raw=np.array([-1.,.044,.044]);physical=np.array([-1.,.044375,.044])
    frames=forward_kinematics(r,physical)
    result=auxiliary_finger_positions(r,raw,frames)
    assert np.allclose(result,physical) and np.array_equal(raw,[-1.,.044,.044])
    assert np.allclose(forward_kinematics(r,result)['link7'],frames['link7'])
    assert r['joint_limits'][1]==[0,.044]


@pytest.mark.parametrize('change',['transverse','rotation','nonfinite','nonprismatic'])
def test_reconstruction_does_not_hide_other_geometry_errors(change):
    r=schema();q=[-1.,.04,.04];frames=forward_kinematics(r,q)
    if change=='transverse':frames['link7'][0,3]+=.001
    if change=='rotation':frames['link7'][:3,:3]=np.diag([-1.,-1.,1.])
    if change=='nonfinite':frames['link7'][1,3]=float('nan')
    if change=='nonprismatic':r['joints'][1]['kind']='revolute'
    with pytest.raises(ValueError):auxiliary_finger_positions(r,q,frames)


def test_agent_target_joint_state_is_unchanged():
    r=schema();r['control_owner']='target';q=[-1.,.04,.04]
    assert np.array_equal(auxiliary_finger_positions(r,q,{}),q)


@pytest.mark.parametrize('owner', ['target', 'support'])
@pytest.mark.parametrize('displacement', [-.0004, .0444])
def test_all_physical_finger_collision_coordinates_include_limit_overtravel(owner, displacement):
    from sim.envs.robodojo.collision_state import measured_finger_positions
    r=schema();r['control_owner']=owner
    q=np.array([-1.,np.clip(displacement,0,.044),.04])
    actual=np.array([-1.,displacement,.04])
    result=measured_finger_positions(r,q,forward_kinematics(r,actual))
    assert np.allclose(result,actual)
    assert q[1]==np.clip(displacement,0,.044)
    assert r['joint_limits'][1]==[0,.044]


@pytest.mark.parametrize('change',['transverse','rotation','nonprismatic','arm_joint'])
def test_target_reconstruction_cannot_hide_other_geometry_errors(change):
    from sim.envs.robodojo.collision_state import measured_finger_positions
    r=schema();r['control_owner']='target';q=[-1.,.04,.04]
    frames=forward_kinematics(r,q)
    if change=='transverse':frames['link7'][0,3]+=.001
    if change=='rotation':frames['link7'][:3,:3]=np.diag([-1.,-1.,1.])
    if change=='nonprismatic':r['joints'][1]['kind']='revolute'
    if change=='arm_joint':r['arm_joint_names'].append('joint7')
    with pytest.raises(ValueError):measured_finger_positions(r,q,frames)
