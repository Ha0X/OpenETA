from copy import deepcopy
from types import SimpleNamespace
from test_robodojo_attached_geometry import grasp_scene
from sim.envs.robodojo.support_geometry import protected_actor_pairs
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.checked_control import CheckedController


def actors():
    world, grants = grasp_scene()
    data=deepcopy(world.scene.snapshot)
    r=deepcopy(data['robots'][0]);r.update(control_owner='support',role='support_arm0')
    data['robots'].append(r)
    for part in list(data['colliders'][:2]):
        item=deepcopy(part);item.update(id='aux_'+part['id'],robot_index=1)
        data['colliders'].append(item)
    world.scene=CollisionScene(data)
    return world, grants


def pair_ids(scene):
    return {frozenset((scene.items[i]['id'],scene.items[j]['id'])) for i,j in scene.pairs}


def test_native_contacts_delegated_but_cross_actor_and_dynamic_geometry_retained():
    world,_=actors();pairs=pair_ids(world.scene)
    assert frozenset(('aux_finger7','held')) not in pairs
    assert frozenset(('aux_finger7','aux_finger8')) not in pairs
    assert frozenset(('finger7','aux_finger7')) in pairs
    assert frozenset(('finger7','held')) in pairs
    assert frozenset(('finger7','wall')) in pairs
    # The coincident cross-actor hands remain a real collision.
    assert not world.scene.check([[-1.,0.,0.],[-1.,0.,0.]])['clear']


def test_native_proximity_does_not_create_rigid_load_or_table_constraint():
    world,_=actors()
    c=SimpleNamespace(support_execution=object(),world=world,contact_grants={},operation='wait',recovery=None)
    scene,evidence=CheckedController._prediction_scene(c,[[-1.,0.,0.],[-1.,0.,0.]])
    assert scene is world.scene and evidence==[]
    assert scene.items[2]['robot_index'] is None
    assert frozenset(('held','wall')) not in pair_ids(scene)


def test_target_held_body_still_checked_against_auxiliary_and_environment():
    world,grants=actors()
    c=SimpleNamespace(support_execution=object(),world=world,contact_grants=grants,operation='move',recovery=None)
    scene,evidence=CheckedController._prediction_scene(c,[[-1.,0.,0.],[-1.,0.,0.]])
    assert evidence and scene.items[2]['robot_index']==0
    assert frozenset(('held','aux_finger7')) in pair_ids(scene)
    assert frozenset(('held','wall')) in pair_ids(scene)
    assert frozenset(('held','aux_finger7')) in protected_actor_pairs(scene)
    assert world.scene.items[2]['robot_index'] is None


def test_ownership_changes_do_not_reuse_incompatible_pair_topology():
    world,_=actors();data=deepcopy(world.scene.snapshot)
    data['robots'][1]['control_owner']='target'
    scene=CollisionScene(data,geometry_source=world.scene,geometry_transforms=world.scene.geometry_transforms)
    assert frozenset(('aux_finger7','held')) in pair_ids(scene)


def passive_finger_scene():
    from test_robodojo_collision_geometry import fixture
    from sim.envs.robodojo.scene_geometry import forward_kinematics
    data=fixture(wall_center=100.)
    robot=deepcopy(data['robots'][0])
    robot.update(control_owner='support',role='support_arm0',arm_joint_names=[],
                 finger_links=['tool'],joint_positions=[2.0004])
    robot['body_frames']={k:v.tolist() for k,v in forward_kinematics(robot,[2.0004]).items()}
    data['robots'].append(robot)
    item=deepcopy(data['colliders'][0]);item.update(id='aux_finger',robot_index=1)
    data['colliders'].append(item)
    return CollisionScene(data)


def test_measured_stationary_auxiliary_finger_does_not_block_target_motion():
    scene=passive_finger_scene()
    result=scene.check_path([[-1.],[2.0004]],[[0.],[2.0004]])
    assert result['status']=='clear' and result['samples']>0
    assert scene.snapshot['robots'][1]['joint_limits']==[[-2.,2.]]
    # The same measured geometry still blocks an approaching target robot.
    blocked=scene.check_path([[-1.],[2.0004]],[[2.],[2.0004]])
    assert blocked['status']=='blocked' and blocked['reason']!='joint_limit'


def test_changed_auxiliary_finger_and_target_joint_limits_remain_enforced():
    scene=passive_finger_scene()
    result=scene.check_path([[-1.],[2.0004]],[[0.],[2.0005]])
    assert result['reason']=='joint_limit' and result['robot_index']==1
    assert result['control_owner']=='support' and result['joint_names']==['slide']
    assert scene.check_path([[-1.],[2.0004]],[[3.],[2.0004]])['robot_index']==0
    data=deepcopy(scene.snapshot);data['robots'][1]['arm_joint_names']=['slide']
    assert CollisionScene(data).check_path([[-1.],[2.0004]],[[0.],[2.0004]])['reason']=='joint_limit'


def test_passive_auxiliary_state_exception_rejects_nonfinite_vectors():
    import pytest
    scene=passive_finger_scene()
    with pytest.raises(ValueError,match='Invalid path joint vector'):
        scene.check_path([[-1.],[float('nan')]],[[0.],[float('nan')]])


def test_target_finger_measurement_keeps_geometry_and_changed_command_limits():
    data=deepcopy(passive_finger_scene().snapshot)
    data['robots'][1]['control_owner']='target'
    scene=CollisionScene(data)
    # A measured finger, unlike an arm command, remains at its actual position.
    assert scene.check_path([[-1.],[2.0004]],[[0.],[2.0004]])['status']=='clear'
    collision=scene.check_path([[-1.],[2.0004]],[[2.],[2.0004]])
    assert collision['status']=='blocked' and collision['reason']!='joint_limit'
    changed=scene.check_path([[-1.],[2.0004]],[[0.],[2.0005]])
    assert changed['reason']=='joint_limit' and changed['control_owner']=='target'
    data['robots'][1]['arm_joint_names']=['slide']
    assert CollisionScene(data).check_path([[-1.],[2.0004]],[[0.],[2.0004]])['reason']=='joint_limit'
