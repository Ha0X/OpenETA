from itertools import product
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.contact_bounds import projection_contact_bound


def scene_for(lo,hi,second_kind='convexHull'):
    def shape(name,a,b,kind):
        return {'id':name,'robot_index':None,'body':None,'approximation':kind,
                'vertices':list(product(*zip(a,b))),'triangles':[]}
    return CollisionScene({'robots':[],'unsupported':[],'filtered_pairs':[],
        'colliders':[shape('a',[-.01]*3,[.01]*3,'convexHull'),shape('b',lo,hi,second_kind)]})


@pytest.mark.parametrize('depth',[.0005,.002])
def test_projection_contact_bound_preserves_shallow_and_excessive_penetration(depth):
    scene=scene_for([.01-depth,-.01,-.01],[.03-depth,.01,.01])
    r=Rotation.from_euler('xyz',[.3,.6,.9]).as_matrix();t=np.eye(4);t[:3,:3]=r
    for i in [0,1]:scene.world_transforms[i]=t;scene.set_transform(i,t)
    measured=projection_contact_bound(scene,[],0,1)
    assert measured<=-depth and abs(measured+depth)<2e-7
    assert (measured>=-.001)==(depth<.001)


def test_projection_bound_uses_separating_translation_for_containment():
    scene=scene_for([-.002]*3,[.002]*3)
    assert projection_contact_bound(scene,[],0,1)<-.0119


def test_projection_bound_supports_calibrated_sphere_and_separation():
    scene=scene_for([.0095,-.01,-.01],[.0295,.01,.01],second_kind='sphere')
    assert abs(projection_contact_bound(scene,[],0,1)+.0005)<2e-7
    t=np.eye(4);t[0,3]=.02;scene.world_transforms[1]=t;scene.set_transform(1,t)
    assert .01949<projection_contact_bound(scene,[],0,1)<.01951


def test_failed_fcl_distance_never_becomes_zero_clearance_or_disables_default_guard(monkeypatch):
    scene=scene_for([.0095,-.01,-.01],[.0295,.01,.01])
    monkeypatch.setattr(scene.fcl,'collide',lambda *args:0)
    monkeypatch.setattr(scene.fcl,'distance',lambda *args:float('nan'))
    with pytest.raises(ValueError,match='unavailable'):
        scene.signed_distances([],[('a','b')])
    bound=scene.signed_distances([],[('a','b')],contact_bounds=True)[frozenset(('a','b'))]
    assert -.000501<bound<-.000499
    assert scene.last_contact_distance_method=='conservative_projection_bound_on_fcl_disagreement'
    assert scene.last_contact_distance_methods[frozenset(('a','b'))] == scene.last_contact_distance_method


def test_contact_separation_pruning_is_fresh_and_never_positive_grasp_evidence(monkeypatch):
    scene=scene_for([.02,-.01,-.01],[.04,.01,.01]);pair=('a','b');key=frozenset(pair)
    exact=scene.signed_distances([],[pair])[key]
    calls=[];original=scene.fcl.collide
    monkeypatch.setattr(scene.fcl,'collide',lambda *args:(calls.append(1),original(*args))[1])
    bounded=scene.signed_distances([],[pair],contact_bounds=True,separation_bounds=True)[key]
    assert 0<bounded<=exact and not calls
    assert scene.last_contact_distance_methods[key]=='aabb_separation_lower_bound'
    # Change the actual transform without refreshing broad-phase caches. The
    # optional penetration query must observe the overlap, not stale bounds.
    t=np.eye(4);t[0,3]=-.012;scene.world_transforms[1]=t;scene.set_transform(1,t)
    bounded=scene.signed_distances([],[pair],contact_bounds=True,separation_bounds=True)[key]
    assert calls and bounded<-.001
    assert scene.last_contact_distance_methods[key]=='fcl_signed_distance'
    with pytest.raises(ValueError,match='bounded contact'):
        scene.signed_distances([],[pair],separation_bounds=True)


@pytest.mark.parametrize('offset',[-.03,-.0205,-.0195,-.01,0.,.03])
def test_contact_pruning_matches_exact_penetration_and_verdict(offset):
    scene=scene_for([.03+offset,-.01,-.01],[.05+offset,.01,.01])
    pairs=[('a','b')]
    exact=scene.signed_distances([],pairs,contact_bounds=True)
    bounded=scene.signed_distances([],pairs,contact_bounds=True,separation_bounds=True)
    assert max(0.,-min(exact.values()))==pytest.approx(max(0.,-min(bounded.values())),abs=1e-12)
    assert (min(exact.values())>=-.001)==(min(bounded.values())>=-.001)
