from itertools import product
import base64
import numpy as np
import pytest

from sim.envs.robodojo.voxel_geometry import surface_cover, encode_octomap, cover_collider, CoverUnavailable, fcl_octree


def cube_shell(center, half):
    # Unit cube [-1,1]^3: entire solid containment is intentionally not a hit.
    lo,hi=center-half,center+half
    return bool(np.all(lo<=1) and np.all(hi>=-1) and not (np.all(lo>-1) and np.all(hi<1)))


def test_surface_oracle_fills_closed_interior_but_not_exterior():
    value=surface_cover([[-1,-1,-1],[1,1,1]],cube_shell,resolution=.2)
    points=(value['indices']+.5)*.2
    assert any(np.linalg.norm(p)<.2 for p in points)
    assert abs(points).max()<=1.1+1e-12
    assert value['occupied_cells']>value['surface_cells']


def test_open_slot_is_not_filled_like_convex_hull():
    # U shape, three box surfaces. Through-slot is exterior-connected.
    boxes=[(np.array([-1,-1,-1]),np.array([-.6,1,1])),
           (np.array([.6,-1,-1]),np.array([1,1,1])),
           (np.array([-1,-1,-1]),np.array([1,-.6,1]))]
    def hit(c,h):
        return any(np.all(c-h<=b) and np.all(c+h>=a) and not(np.all(c-h>a) and np.all(c+h<b)) for a,b in boxes)
    value=surface_cover([[-1]*3,[1]*3],lambda c,h:bool(hit(c,h)),resolution=.1)
    points=(value['indices']+.5)*.1
    assert not np.any(np.linalg.norm(points,axis=1)<.3)
    assert np.any(np.linalg.norm(points-[-.8,0,0],axis=1)<.1)


@pytest.mark.parametrize('kw',[{'max_queries':1},{'max_cells':2}])
def test_exhaustion_never_emits_partial_cover(kw):
    with pytest.raises(CoverUnavailable,match='budget'):
        surface_cover([[-1]*3,[1]*3],cube_shell,resolution=.1,**kw)


def test_unknown_empty_and_bad_domain_are_not_admitted():
    for query in (lambda c,h:None,lambda c,h:False,lambda c,h:True):
        with pytest.raises(CoverUnavailable):surface_cover([[-1]*3,[1]*3],query,resolution=1)
    with pytest.raises(CoverUnavailable):surface_cover([[0]*3,[0]*3],cube_shell)


def test_octomap_real_fcl_collision_distance_and_transform():
    import fcl
    keys=np.array([[-1,0,0],[1,0,0]])
    shape=fcl_octree(.1,encode_octomap(keys))
    object_=fcl.CollisionObject(shape)
    def hit(p,r=.01):
        sphere=fcl.CollisionObject(fcl.Sphere(r),fcl.Transform(np.array(p,dtype=float)))
        result=fcl.CollisionResult()
        fcl.collide(object_,sphere,fcl.CollisionRequest(),result)
        distance=fcl.distance(object_,sphere,fcl.DistanceRequest(),fcl.DistanceResult()) if not result.is_collision else 0
        return result.is_collision,distance
    assert hit([-.05,.05,.05])[0]
    assert hit([.15,.05,.05])[0]
    clear,distance=hit([.05,.05,.05])
    assert not clear and distance==pytest.approx(.04)
    assert not hit([.05,.3,.05])[0]
    object_.setTransform(fcl.Transform(np.array([0.,1.,0.])))
    assert not hit([-.05,.05,.05])[0]
    assert hit([-.05,1.05,.05])[0]


def test_collider_retains_immutable_source_and_outer_bounds():
    cover={'indices':np.array([[-1,0,0],[1,0,0]]),'resolution_m':.1,'queries':12}
    item=cover_collider(cover,mesh_path='/native/model',collider_path='/native/model',
                        reference_transform=np.eye(4),native_approximation='meshSimplification')
    assert item['source_prim_path']=='/native/model'
    assert base64.b64decode(item['octomap_binary'])==encode_octomap(cover['indices'])
    assert np.allclose(np.asarray(item['vertices']).min(0),[-.1,0,0])
    assert np.allclose(np.asarray(item['vertices']).max(0),[.2,.1,.1])


def scene_with_voxel_wall():
    from test_robodojo_collision_geometry import fixture
    from sim.envs.robodojo.collision_geometry import CollisionScene
    data=fixture(start=-.1)
    keys=np.array(list(product(range(-5,5),repeat=3)),dtype=int)
    data['colliders'][1]=cover_collider({'indices':keys,'resolution_m':.01},mesh_path='world',
        collider_path='world',reference_transform=np.eye(4),native_approximation='sdf')
    return CollisionScene(data)


def test_voxel_cover_enters_existing_swept_collision_and_containment_checks():
    scene=scene_with_voxel_wall()
    assert scene.check([[-1.]])['clear']
    assert not scene.check([[0.]])['clear']
    assert scene.check_path([[-1.]],[[1.]])['status']=='blocked'
    assert scene.check_path([[-1.]],[[-.7]])['status']=='clear'


def test_contact_bound_is_full_union_separating_translation_without_mutation():
    scene=scene_with_voxel_wall()
    q=[[-.0995]]
    before=scene.check(q)
    assert not before['clear']
    with pytest.raises(ValueError,match='not exact signed proximity'):
        scene.signed_distances(q,[('finger','world')])
    bound=scene.signed_distances(q,[('finger','world')],contact_bounds=True)[frozenset(('finger','world'))]
    assert -.00051 < bound < -.0005
    assert not scene.check(q)['clear']
    assert scene.last_contact_distance_method=='verified_translation_upper_bound_on_native_voxel_cover'
    # A wholly contained query shape must use union depth, not a single cell.
    bound=scene.signed_distances([[0.]],[('finger','world')],contact_bounds=True)[frozenset(('finger','world'))]
    assert bound < -.0999


@pytest.mark.parametrize('signed',[True,False])
def test_octomap_char_abi_works_on_x86_and_arm(monkeypatch,signed):
    import sys
    from types import SimpleNamespace
    seen=[]
    def make(resolution,data):
        numbers=list(data)
        if signed and max(numbers)>127:raise OverflowError('value too large')
        if not signed and min(numbers)<0:raise OverflowError('negative value')
        seen.append(bytes(n%256 for n in numbers))
        return 'shape'
    monkeypatch.setitem(sys.modules,'fcl',SimpleNamespace(OcTree=make))
    assert fcl_octree(.001,b'\xff\x02')=='shape'
    assert seen==[b'\xff\x02']


def test_native_query_domain_does_not_clip_source_vertices_to_cooked_bounds():
    from sim.envs.robodojo.voxel_geometry import native_query_domain
    cooked=[[-.01,-.02,-.003],[.01,.02,.003]]
    points=np.array([[-.015,-.01,-.002],[.015,.01,.002]])
    transform=np.eye(4);transform[:3,:3]=[[0,-2,0],[3,0,0],[0,0,.5]];transform[:3,3]=[.2,.3,.4]
    domain=native_query_domain(cooked,points,transform)
    assert np.allclose(domain,[[.16,.255,.3985],[.24,.345,.4015]])
    expanded=native_query_domain(cooked,points,transform,sdf_margin=.01)
    assert np.all(expanded[0]<domain[0]) and np.all(expanded[1]>domain[1])
    with pytest.raises(CoverUnavailable):native_query_domain(cooked,points,transform,sdf_margin=float('nan'))
