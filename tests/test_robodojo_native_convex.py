from itertools import product
import copy
import numpy as np
import pytest
from scipy.spatial import ConvexHull

from sim.envs.robodojo.native_convex import decode_convex_data, world_convex_colliders
from sim.envs.robodojo.collision_geometry import CollisionScene


def cube(lo, hi):
    vertices = np.array(list(product(*zip(lo, hi))))
    hull = ConvexHull(vertices)
    faces = hull.simplices
    return {'vertices': vertices.tolist(), 'indices': faces.ravel().tolist(),
            'polygons': [{'index_base': 3*i, 'num_vertices': 3, 'plane': hull.equations[i].tolist()}
                         for i in range(len(faces))]}


def test_cooked_planes_define_surface_even_when_native_vertices_are_quantized():
    value = cube([-.01]*3, [.01]*3)
    # Plane-defined physics surface is 0.2mm outside its quantized vertices.
    for face in value['polygons']:
        face['plane'][3] -= .0002
    points, _ = decode_convex_data(value)
    assert np.allclose(points.min(axis=0), -.0102)
    assert np.allclose(points.max(axis=0), .0102)
    item = world_convex_colliders([(points, [])], '/a', '/a', np.eye(4))[0]
    probe = world_convex_colliders([decode_convex_data(cube([.0101,-.001,-.001],[.011,.001,.001]))],
                                  '/b', '/b', np.eye(4))[0]
    scene = CollisionScene({'robots': [], 'colliders': [item,probe], 'unsupported': [], 'filtered_pairs': []})
    assert min(scene.signed_distances([], [('/a','/b')]).values()) < 0


@pytest.mark.parametrize('damage', ['zero_normal', 'nan_plane', 'open_solid', 'empty_solid'])
def test_invalid_native_planes_fail_closed(damage):
    value = cube([-1]*3,[1]*3)
    if damage == 'zero_normal': value['polygons'][0]['plane'] = [0,0,0,1]
    if damage == 'nan_plane': value['polygons'][0]['plane'][3] = float('nan')
    if damage == 'open_solid':
        for p in value['polygons']:
            if p['plane'][0] < -.5: p['plane'][0] = 1.
    if damage == 'empty_solid':
        for p in value['polygons']: p['plane'][3] = 1.
    with pytest.raises(ValueError): decode_convex_data(value)


def test_native_parts_preserve_cavity_and_still_detect_wall_penetration():
    # Two walls and a base, with a 2cm probe inside the open cavity.
    parts = [decode_convex_data(cube(lo, hi)) for lo, hi in [
        ([-.1,-.1,-.02],[.1,.1,0]),
        ([-.1,-.1,0],[-.08,.1,.1]),
        ([.08,-.1,0],[.1,.1,.1])]]
    items = world_convex_colliders(parts, '/container/mesh', '/container/mesh', np.eye(4))
    probe = world_convex_colliders([decode_convex_data(cube([-.01,-.01,.03],[.01,.01,.05]))],
                                  '/probe/mesh', '/probe/mesh', np.eye(4))[0]
    scene = CollisionScene({'robots': [], 'colliders': items+[probe], 'unsupported': [], 'filtered_pairs': []})
    pairs = [(item['id'], probe['id']) for item in items]
    assert min(scene.signed_distances([], pairs).values()) > 0
    enclosing = copy.deepcopy(items[0])
    enclosing['vertices'] = np.concatenate([item['vertices'] for item in items]).tolist()
    old = CollisionScene({'robots': [], 'colliders': [enclosing,probe], 'unsupported': [], 'filtered_pairs': []})
    assert old.signed_distances([], [(enclosing['id'],probe['id'])])[frozenset((enclosing['id'],probe['id']))] < -.001
    shift = np.eye(4); shift[0,3] = .075
    scene.world_transforms[3] = shift; scene.set_transform(3,shift)
    assert min(scene.signed_distances([], pairs).values()) < -.001


def test_native_part_sources_and_scaled_world_coordinates_are_explicit():
    parts = [decode_convex_data(cube([-1,-1,-1],[1,1,1]))]*2
    transform = np.eye(4); transform[:3,:3] = np.diag([.1,.2,.3]); transform[:3,3] = [1,2,3]
    items = world_convex_colliders(parts, '/source/mesh', '/source', transform)
    assert len({item['id'] for item in items}) == 2
    for item in items:
        assert item['source_prim_path']=='/source/mesh' and item['collider_path']=='/source'
        assert item['native_part_count']==2 and item['geometry_source']=='physx_cooked_convex_part'
        assert np.allclose(np.min(item['vertices'],axis=0),[.9,1.8,2.7])
    single = world_convex_colliders(parts[:1], '/source/mesh', '/source', transform)
    assert single[0]['id']=='/source/mesh'


@pytest.mark.parametrize('damage', ['nan','bad_index','missing_polygon','overlapping_polygon'])
def test_invalid_cooked_data_is_not_silently_replaced_with_an_outer_hull(damage):
    value = cube([-1,-1,-1],[1,1,1])
    if damage=='nan': value['vertices'][0][0]=float('nan')
    if damage=='bad_index': value['indices'][0]=100
    if damage=='missing_polygon': value['polygons'].pop()
    if damage=='overlapping_polygon': value['polygons'][1]['index_base']=0
    with pytest.raises(ValueError): decode_convex_data(value)
    with pytest.raises(ValueError,match='no parts'): world_convex_colliders([], '/mesh', '/mesh', np.eye(4))


def test_cooked_single_hull_keeps_native_provenance_and_rejects_multiple_parts():
    from sim.envs.robodojo.native_convex import world_convex_colliders
    part=(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1.]]),[[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    item=world_convex_colliders([part],'/asset/mesh','/asset/mesh',np.eye(4),native_approximation='convexHull')[0]
    assert item['native_approximation']=='convexHull'
    assert item['geometry_source']=='physx_cooked_convex_part'
    with pytest.raises(ValueError,match='exactly one'):
        world_convex_colliders([part,part],'/asset/mesh','/asset/mesh',np.eye(4),native_approximation='convexHull')


@pytest.mark.parametrize('owners,expected', [([], '/parent'),(['/other'],'/parent'),(['/parent','/other'],'/parent'),(['/parent'],None)])
def test_native_ownership_requires_positive_consistent_hit(owners,expected):
    from sim.envs.robodojo.native_convex import require_matching_owner
    with pytest.raises(ValueError,match='ownership'):require_matching_owner(owners,expected)


def test_multiple_convex_parts_may_share_one_verified_physical_owner():
    from sim.envs.robodojo.native_convex import require_matching_owner
    assert require_matching_owner(['/parent','/parent'],'/parent')=='/parent'
