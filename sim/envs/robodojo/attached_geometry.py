"""Private conditional rigid-grasp geometry, never a grasp-success oracle.

Only agent-authorized bodies near BOTH measured fingers are candidates. The
prediction changes collision kinematics, not PhysX poses or collision flags.
"""
from copy import deepcopy
import time

import numpy as np

from .collision_geometry import CollisionScene
from .scene_geometry import array, forward_kinematics, transform_vertices


def attachment_frames(world, positions, grants, evidence):
    """Measured object-relative-to-wrist frames for slip monitoring."""
    frames = [forward_kinematics(r, q) for r, q in zip(world.scene.snapshot['robots'], positions)]
    result = {}
    for item in evidence:
        role = item['role']
        index = next(i for i,r in enumerate(world.scene.snapshot['robots']) if r['role'] == role)
        grant = grants[role]; collider = grant['indices'][0]
        # A decomposed body's reference must not depend on which native hull
        # happens to be first. All parts share the same rigid-body transform.
        vertices = np.concatenate([world.scene.vertices[i] for i in grant['indices']])
        center = np.eye(4)
        center[:3, 3] = (vertices.min(axis=0)+vertices.max(axis=0))/2
        result[role] = np.linalg.inv(frames[index][world.scene.snapshot['robots'][index].get('tool_link', 'link6')]) @ world.scene.world_transforms[collider] @ center
    return result


def attached_scene(world, positions, grants):
    began = time.perf_counter()
    result = _attached_scene(world, positions, grants)
    stats = getattr(world, 'attachment_performance', None)
    if stats is None:
        stats = world.attachment_performance = {'calls':0, 'elapsed_s':0., 'reused_shape_views':0}
    stats['calls'] += 1
    stats['elapsed_s'] += time.perf_counter()-began
    stats['reused_shape_views'] += int(result[0] is not world.scene)
    stats.setdefault('reused_pair_topologies', 0)
    stats['reused_pair_topologies'] += int(result[0] is not world.scene and result[0].pair_topology_reused)
    return result


def _attached_scene(world, positions, grants):
    scene = world.scene
    if not grants:
        return scene, []
    attachments, claimed, closed_chains = [], {}, []
    initial_frames = [forward_kinematics(r, q) for r, q in zip(scene.snapshot['robots'], positions)]
    candidates = []
    for role, grant in grants.items():
        if grant.get('kind') == 'cloth_patch':
            continue  # Material contact never becomes a rigid attachment.
        index = next(i for i, r in enumerate(scene.snapshot['robots']) if r['role'] == role)
        schema = scene.snapshot['robots'][index]
        finger_slots = [k for k, n in enumerate(schema['joint_names']) if n not in schema['arm_joint_names']]
        if not finger_slots:
            continue
        openness = [(positions[index][k]-schema['joint_limits'][k][0]) /
                    (schema['joint_limits'][k][1]-schema['joint_limits'][k][0]) for k in finger_slots]
        if min(openness) > .95:
            continue
        groups = []
        for item in scene.items:
            if item['robot_index'] != index or item['body'] not in schema.get('finger_links', ('link7', 'link8')):
                continue
            pairs = [(item['id'], scene.items[j]['id']) for j in grant['indices']]
            groups.append((item['body'], pairs))
        candidates.append((index, grant, groups))
    all_pairs = [pair for _, _, groups in candidates for _, pairs in groups for pair in pairs]
    # One FK/object refresh for the same measured state, still exact FCL finger
    # distances. Never reuse decisions across physics steps or use broad-phase
    # lower bounds as affirmative two-finger proximity evidence.
    try:
        batch = scene.signed_distances(positions, all_pairs) if all_pairs else {}
    except ValueError:
        batch = None  # Preserve the original per-finger unknown isolation.
    for index, grant, groups in candidates:
        touching = set()
        for body, pairs in groups:
            if batch is None:
                try:
                    distances = scene.signed_distances(positions, pairs)
                except ValueError:
                    continue
            else:
                distances = {frozenset(pair): batch[frozenset(pair)] for pair in pairs}
            if min(distances.values(), default=np.inf) <= .002:
                touching.add(body)
        if touching != set(scene.snapshot['robots'][index].get('finger_links', ('link7', 'link8'))):
            continue
        if grant['root'] in claimed:
            owner = claimed[grant['root']]
            relative = np.linalg.inv(initial_frames[owner][scene.snapshot['robots'][owner].get('tool_link', 'link6')]) @ initial_frames[index][scene.snapshot['robots'][index].get('tool_link', 'link6')]
            closed_chains.append({'owner': owner, 'follower': index, 'relative_pose': relative.tolist(),
                                  'owner_link': scene.snapshot['robots'][owner].get('tool_link', 'link6'),
                                  'follower_link': scene.snapshot['robots'][index].get('tool_link', 'link6')})
            continue
        claimed[grant['root']] = index
        attachments.append((index, grant['indices']))
    if not attachments:
        return scene, []
    # Do not deep-copy mesh buffers or rebuild hulls at every physics step.
    # This is a fresh kinematic view, not a cached attachment decision: finger
    # evidence, world transforms, ownership and closed chains are re-evaluated.
    snapshot = dict(scene.snapshot)
    snapshot['robots'] = deepcopy(scene.snapshot['robots'])
    snapshot['colliders'] = [{**item,'vertices':scene.vertices[i]} for i,item in enumerate(scene.items)]
    geometry_transforms = [t.copy() for t in scene.geometry_transforms]
    snapshot['closed_chain_constraints'] = closed_chains
    frames = [forward_kinematics(r, q) for r, q in zip(snapshot['robots'], positions)]
    for r, q, f in zip(snapshot['robots'], positions, frames):
        r['joint_positions'] = array(q).tolist()
        r['body_frames'] = {name: value.tolist() for name, value in f.items()}
    for i, item in enumerate(snapshot['colliders']):
        if item['robot_index'] is None:
            t = scene.world_transforms[i]
            item['vertices'] = transform_vertices(scene.vertices[i], t, item['approximation'])
            geometry_transforms[i] = t @ geometry_transforms[i]
    public = []
    for index, indices in attachments:
        body = f'held_geometry_{min(indices)}'
        robot = snapshot['robots'][index]
        tool_link = robot.get('tool_link', 'link6')
        reference = frames[index][tool_link]
        inverse = np.linalg.inv(reference)
        robot['body_frames'][body] = reference.tolist()
        robot['joints'].append({'name': body+'_fixed', 'parent': tool_link, 'child': body,
            'kind': 'fixed', 'axis': 'X', 'parent_frame': np.eye(4).tolist(),
            'child_frame': np.eye(4).tolist(), 'collision_enabled': True})
        for i in indices:
            item = snapshot['colliders'][i]
            item.update(robot_index=index, body=body,
                vertices=transform_vertices(array(item['vertices']), inverse, item['approximation']))
            geometry_transforms[i] = inverse @ geometry_transforms[i]
        public.append({'role': robot['role'], 'contact_id': f'contact_{min(indices)}',
            'mode': 'conditional_rigid_attachment', 'grasp_confirmed': False,
            'evidence': 'authorized_body_with_both_fingers_within_2mm',
            'assumption': 'body_follows_measured_wrist; actual_physics_is_monitored'})
    for constraint in closed_chains:
        public.append({'role': snapshot['robots'][constraint['follower']]['role'],
            'mode': 'shared_body_relative_wrist_constraint', 'grasp_confirmed': False,
            'translation_tolerance_m': .003, 'rotation_tolerance_rad': .03})
    return CollisionScene(snapshot, geometry_source=scene, geometry_transforms=geometry_transforms), public
