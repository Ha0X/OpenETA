from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim.envs.robodojo.scene_geometry import collision_envelope, transform_vertices
from sim.envs.robodojo.collision_geometry import CollisionScene
from test_robodojo_collision_geometry import fixture


def test_sphere_center_scale_and_rotation_are_preserved():
    points = np.vstack([np.eye(3), -np.eye(3)])*.005+[0,0,.005]
    v, faces, kind = collision_envelope(points, [], 'boundingSphere')
    transform = np.eye(4); transform[:3,:3] = Rotation.from_euler('xyz',[.3,.4,.5]).as_matrix()*2
    transform[:3,3] = [.1,.2,.3]
    world = transform_vertices(v, transform, kind)
    assert np.allclose(world.mean(0), transform[:3,:3]@[0,0,.005]+transform[:3,3])
    assert np.allclose(world.max(0)-world.min(0), .02, atol=1e-8)
    transform[0,:3] *= 2
    with pytest.raises(ValueError, match='uniform scale'):
        transform_vertices(v, transform, kind)


def test_arbitrary_bounding_sphere_mesh_is_not_guessed():
    points = np.array(list(product([-1,1],[-.5,.5],[-.2,.2])))
    with pytest.raises(ValueError, match='verified spherical mesh'):
        collision_envelope(points, [], 'boundingSphere')


def test_sphere_collision_uses_analytic_radius_and_offset_not_box_corners():
    data = fixture(start=-.1)
    sphere = data['colliders'][1]
    sphere.update(approximation='sphere', vertices=np.array(list(product([-.05,.05],repeat=3))).tolist())
    scene = CollisionScene(data)
    assert not scene.check([[-.08]])['clear']
    assert scene.check([[-.12]])['clear']
    shift = np.eye(4); shift[0,3] = 1.
    scene.set_transform(1,shift)
    assert np.allclose(scene.objects[1].getTranslation(),[1,0,0])


def test_measured_sphere_calibration_updates_reused_radius_and_center():
    data=fixture(start=-.1)
    data['colliders'][1].update(approximation='sphere',vertices=np.array(list(product([-.05,.05],repeat=3))).tolist())
    scene=CollisionScene(data)
    scene.replace_world_sphere(1,np.array([.2,0,0]),.02)
    view=CollisionScene(scene.snapshot,geometry_source=scene,geometry_transforms=scene.geometry_transforms)
    oracle=CollisionScene(scene.snapshot)
    assert np.allclose(scene.objects[1].getTranslation(),[.2,0,0])
    pair=[(data['colliders'][0]['id'],data['colliders'][1]['id'])]
    for q in [-.1,.15,.17]:
        a=view.signed_distances([[q]],pair);b=oracle.signed_distances([[q]],pair)
        assert abs(next(iter(a.values()))-next(iter(b.values())))<1e-8
