from copy import deepcopy

import numpy as np

from test_robodojo_attached_geometry import grasp_scene
from sim.envs.robodojo.attached_geometry import attached_scene
from sim.envs.robodojo.collision_geometry import CollisionScene


def test_reused_topology_keeps_fresh_geometry_and_distances():
    world, grants = grasp_scene()
    first, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    before = first.signed_distances([[-1.,0.,0.]], [('held','wall')])
    transform = np.eye(4); transform[0,3] = -.25
    world.scene.world_transforms[3] = transform
    world.scene.set_transform(3, transform)
    second, evidence = attached_scene(world, [[-1.,0.,0.]], grants)
    assert evidence and second.pair_topology_reused
    oracle = CollisionScene(deepcopy(second.snapshot))
    assert second.pairs == oracle.pairs
    assert all(a is not b for a,b in zip(first.objects,second.objects))
    after = second.signed_distances([[-1.,0.,0.]], [('held','wall')])
    assert next(iter(after.values())) < next(iter(before.values())) - .2
    for x in [-1.,-.7,-.35]:
        actual = second.check([[x,0.,0.]], allowed_pairs=grants['left']['pairs'])
        expected = oracle.check([[x,0.,0.]], allowed_pairs=grants['left']['pairs'])
        assert actual['clear'] == expected['clear']
        assert abs(actual['clearance_m']-expected['clearance_m']) < 1e-7
    first.pairs.clear()
    third, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    assert third.pairs == oracle.pairs


def test_filter_policy_changes_invalidate_topology():
    world, grants = grasp_scene()
    first, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    assert (3,2) in first.pairs
    world.scene.snapshot['filtered_pairs'] = [('held','wall')]
    second, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    assert not second.pair_topology_reused and (3,2) not in second.pairs
    world.scene.snapshot['filtered_pairs'] = []
    third, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    assert third.pair_topology_reused and (3,2) in third.pairs


def test_body_assignment_and_adjacent_link_rules_invalidate_topology():
    world, grants = grasp_scene()
    source = world.scene
    a, _ = attached_scene(world, [[-1.,0.,0.]], grants)
    snapshot = deepcopy(a.snapshot)
    snapshot['colliders'][2]['body'] = 'link6'
    # Assign to the parent of the fingers: the existing disabled adjacent-link
    # collisions must now apply, unlike the independent held-body topology.
    b = CollisionScene(snapshot, geometry_source=source, geometry_transforms=a.geometry_transforms)
    assert not b.pair_topology_reused and (2,0) not in b.pairs and (2,0) in a.pairs
    next(j for j in snapshot['robots'][0]['joints'] if j['name']=='joint7')['collision_enabled'] = True
    c = CollisionScene(snapshot, geometry_source=source, geometry_transforms=a.geometry_transforms)
    assert not c.pair_topology_reused and (2,0) in c.pairs


def test_topology_cache_is_bounded_and_never_caches_grasp_evidence():
    world, grants = grasp_scene()
    for n in range(12):
        world.scene.snapshot['filtered_pairs'] = [(f'unused_{n}', 'wall')]
        attached_scene(world,[[-1.,0.,0.]],grants)
    assert len(world.scene._pair_topologies) <= 8
    view, evidence = attached_scene(world,[[-1.,.03,0.]],grants)
    assert view is world.scene and not evidence
