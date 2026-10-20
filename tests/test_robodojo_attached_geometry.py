from itertools import product
from types import SimpleNamespace

import numpy as np

from sim.envs.robodojo.attached_geometry import attached_scene
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.scene_geometry import forward_kinematics


def grasp_scene():
    eye = np.eye(4)
    robot = {'role': 'left', 'base_link': 'base', 'joint_names': ['slide', 'joint7', 'joint8'],
        'arm_joint_names': ['slide'], 'joint_positions': [-1., 0., 0.],
        'joint_limits': [[-2, 2], [0, .044], [0, .044]],
        'body_frames': {n: eye.tolist() for n in ('base', 'link6', 'link7', 'link8')}, 'joints': []}
    for name, parent, child, offset, axis in [('slide', 'base', 'link6', 0, 'X'),
            ('joint7', 'link6', 'link7', .04, 'Y'), ('joint8', 'link6', 'link8', -.04, 'Y')]:
        frame = eye.copy(); frame[1, 3] = offset
        robot['joints'].append({'name': name, 'parent': parent, 'child': child, 'kind': 'prismatic',
            'axis': axis, 'parent_frame': frame.tolist(), 'child_frame': eye.tolist(), 'collision_enabled': False})
    robot['body_frames'] = {n: t.tolist() for n,t in forward_kinematics(robot, [-1.,0.,0.]).items()}
    def box(name, lo, hi, index, body):
        return {'id': name, 'robot_index': index, 'body': body, 'approximation': 'convexHull',
            'vertices': list(product(*zip(lo,hi))), 'triangles': []}
    parts = [box('finger7', [-.01]*3, [.01]*3, 0, 'link7'),
             box('finger8', [-.01]*3, [.01]*3, 0, 'link8'),
             box('held', [-1.2,-.03,-.03], [-.8,.03,.03], None,None),
             box('wall', [-.20,-.05,-.05], [-.10,.05,.05], None,None)]
    scene = CollisionScene({'robots': [robot], 'colliders': parts, 'unsupported': [], 'filtered_pairs': []})
    grants = {'left': {'root': 'body', 'indices': [2], 'pairs': [('finger7','held'),('finger8','held')]}}
    return SimpleNamespace(scene=scene), grants


def test_attachment_checks_object_world_sweep_without_mutating_physics_snapshot():
    world, grants = grasp_scene()
    start = [[-1.,0.,0.]]
    scene, evidence = attached_scene(world, start, grants)
    assert len(evidence) == 1 and evidence[0]['grasp_confirmed'] is False
    assert world.scene.items[2]['robot_index'] is None
    assert scene.items[2]['robot_index'] == 0
    allowed = grants['left']['pairs']
    assert scene.check_path(start, [[-.7,0.,0.]], allowed_pairs=allowed)['status'] == 'clear'
    assert scene.check_path(start, [[-.35,0.,0.]], allowed_pairs=allowed)['status'] != 'clear'
    # Original checker sees a stationary world object and misses this sweep.
    assert world.scene.check_path(start, [[-.35,0.,0.]], allowed_pairs=allowed)['status'] == 'clear'


def test_authorization_without_two_finger_evidence_never_attaches():
    world, grants = grasp_scene()
    scene, evidence = attached_scene(world, [[-1.,.02,0.]], grants)
    assert scene is world.scene and evidence == []


def test_no_grant_never_exposes_or_attaches_world_objects():
    world, _ = grasp_scene()
    scene, evidence = attached_scene(world, [[-1.,0.,0.]], {})
    assert scene is world.scene and not evidence


def test_finger_queries_batched_without_stale_evidence(monkeypatch):
    world,grants=grasp_scene();original=world.scene.signed_distances;calls=[]
    def measured(q,pairs,**kwargs):
        calls.append((list(pairs),kwargs))
        return original(q,pairs,**kwargs)
    monkeypatch.setattr(world.scene,'signed_distances',measured)
    _,evidence=attached_scene(world,[[-1.,0.,0.]],grants)
    assert evidence and len(calls)==1 and len(calls[0][0])==2 and not calls[0][1]
    _,evidence=attached_scene(world,[[-1.,.02,0.]],grants)
    assert not evidence and len(calls)==2


def test_batch_query_failure_preserves_individual_finger_unknowns(monkeypatch):
    world,grants=grasp_scene();original=world.scene.signed_distances
    def failure(q,pairs,**kwargs):
        if len(pairs)>1 or pairs[0][0]=='finger7':raise ValueError('distance unavailable')
        return original(q,pairs,**kwargs)
    monkeypatch.setattr(world.scene,'signed_distances',failure)
    _,evidence=attached_scene(world,[[-1.,0.,0.]],grants)
    assert not evidence


def test_attachment_reference_is_invariant_to_native_convex_part_order():
    from sim.envs.robodojo.attached_geometry import attachment_frames
    world, grants = grasp_scene()
    snapshot = world.scene.snapshot
    part = dict(snapshot['colliders'][2], id='held_second')
    snapshot['colliders'][2]['vertices'] = list(product(*zip([-1.2,-.03,-.03],[-1.,.03,.03])))
    part['vertices'] = list(product(*zip([-1.,-.03,-.03],[-.8,.03,.03])))
    snapshot['colliders'].append(part)
    world.scene = CollisionScene(snapshot)
    grants['left']['indices'] = [2,4]
    a = attachment_frames(world,[[-1.,0.,0.]],grants,[{'role':'left'}])['left']
    grants['left']['indices'] = [4,2]
    b = attachment_frames(world,[[-1.,0.,0.]],grants,[{'role':'left'}])['left']
    assert np.allclose(a,b) and np.allclose(a,np.eye(4))


def test_reused_shapes_match_rebuilt_oracle_and_do_not_rebuild_hulls(monkeypatch):
    import sim.envs.robodojo.collision_geometry as geometry
    from scipy.spatial.transform import Rotation
    world, grants = grasp_scene()
    # Include a moved/rotated world body. Reuse must not freeze the old pose.
    transform = np.eye(4)
    transform[:3,:3] = Rotation.from_euler('x', .15).as_matrix()
    transform[2,3] = .001
    world.scene.world_transforms[2] = transform
    world.scene.set_transform(2, transform)
    calls = []
    original = geometry._geometry
    monkeypatch.setattr(geometry, '_geometry', lambda item:(calls.append(item['id']),original(item))[1])
    scene, evidence = attached_scene(world, [[-1.,0.,0.]], grants)
    assert evidence and calls == []
    assert scene.geometries is world.scene.geometries
    assert all(a is not b for a,b in zip(scene.objects,world.scene.objects))
    oracle = CollisionScene(scene.snapshot)
    pairs = [('held','wall'),('held','finger7')]
    for x in [-1.,-.7,-.35]:
        measured = scene.signed_distances([[x,0.,0.]], pairs)
        expected = oracle.signed_distances([[x,0.,0.]], pairs)
        assert all(abs(measured[k]-expected[k])<1e-7 for k in expected)
    # Fresh call must re-check finger evidence; shared geometry is not a grasp cache.
    scene2, evidence2 = attached_scene(world, [[-1.,.03,0.]], grants)
    assert scene2 is world.scene and not evidence2


def test_translated_sphere_reuse_matches_rebuilt_geometry():
    world, grants = grasp_scene()
    # Sphere centre is encoded in vertices rather than the FCL radius shape.
    snapshot = world.scene.snapshot
    snapshot['colliders'][3].update(approximation='sphere',
        vertices=[[-.2,-.05,-.05],[-.1,.05,.05]])
    world.scene = CollisionScene(snapshot)
    t=np.eye(4);t[0,3]=.02
    world.scene.world_transforms[3]=t
    scene, _ = attached_scene(world,[[-1.,0.,0.]],grants)
    oracle=CollisionScene(scene.snapshot)
    pairs=[('held','wall')]
    for x in [-1.,-.6]:
        a=scene.signed_distances([[x,0.,0.]],pairs)
        b=oracle.signed_distances([[x,0.,0.]],pairs)
        assert abs(next(iter(a.values()))-next(iter(b.values())))<1e-7


def test_franka_attachment_uses_configured_hand_and_finger_frames():
    world, grants = grasp_scene()
    data = world.scene.snapshot
    names = {'link6':'panda_hand','link7':'panda_leftfinger','link8':'panda_rightfinger'}
    robot=data['robots'][0]
    robot['tool_link']='panda_hand';robot['finger_links']=['panda_leftfinger','panda_rightfinger']
    robot['body_frames']={names.get(k,k):v for k,v in robot['body_frames'].items()}
    for joint in robot['joints']:
        joint['parent']=names.get(joint['parent'],joint['parent'])
        joint['child']=names.get(joint['child'],joint['child'])
    for part in data['colliders']:part['body']=names.get(part['body'],part['body'])
    world.scene=CollisionScene(data)
    scene,evidence=attached_scene(world,[[-1.,0.,0.]],grants)
    assert evidence and scene.items[2]['robot_index']==0
    assert scene.snapshot['robots'][0]['joints'][-1]['parent']=='panda_hand'
    assert scene.check_path([[-1.,0.,0.]], [[-.35,0.,0.]], allowed_pairs=grants['left']['pairs'])['status']!='clear'
