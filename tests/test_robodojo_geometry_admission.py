import importlib.util
from pathlib import Path
import numpy as np
import pytest


def load(name):
    import importlib
    return importlib.import_module('sim.envs.robodojo.'+name)


def test_primitive_is_present_in_collision_oracle_with_scale_and_rotation():
    import fcl
    from sim.envs.robodojo.collision_geometry import _geometry
    m=load('primitive_geometry')
    angle=np.pi/4
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    t=np.eye(4);t[:3,:3]=rotation@np.diag([.1,.2,.3]);t[:3,3]=[.5,.5,.5]
    m.validate_cube_transform(t)
    vertices=m.cube_vertices(1)@t[:3,:3].T+t[:3,3]
    box=fcl.CollisionObject(_geometry({'approximation':'convexHull','vertices':vertices,'triangles':[]}))
    def collision(local):
        center=t[:3,:3]@local+t[:3,3]
        sphere=fcl.CollisionObject(fcl.Sphere(.001),fcl.Transform(center))
        return fcl.collide(box,sphere,fcl.CollisionRequest(),fcl.CollisionResult())
    assert collision([.49,0,0])
    assert not collision([.6,0,0])
    t[0,1]+=.04
    with pytest.raises(ValueError,match='shear'):m.validate_cube_transform(t)
    for bad in (0,-1,float('nan')):
        with pytest.raises(ValueError):m.cube_vertices(bad)


def test_error_retains_reason_without_object_or_filesystem_identity():
    m=load('public_errors')
    original="Incomplete collision coverage: [{'path': '/World/envs/env_0/rigid/key/key_1/model', 'reason': 'non_mesh_collider'}]"
    result=m.public_error_message(ValueError(original))
    assert 'non_mesh_collider' in result and 'private_geometry_' in result
    assert '/World' not in result and 'key_1' not in result
    assert m.public_error_message(ValueError('Invalid /home/example/private/data.usd'))=='Invalid '+m.public_error_message(ValueError('/home/example/private/data.usd'))
    assert m.public_error_message(ValueError('Stale image: observe again before marking'))=='Stale image: observe again before marking'


def test_stability_diagnostics_preserve_native_result_and_restore_reader():
    from types import SimpleNamespace
    m=load('stability_diagnostics')
    class Layout:
        def get_instance_pose(self,*,env_idx,inst_name):return np.array([0.,0.,.8]),np.array([1.,0.,0.,0.])
        def check_layout_stability(self,env):
            self.get_instance_pose(env_idx=0,inst_name='private_actor')
            if env=='raise':raise RuntimeError('native failure')
            return False,[0]
    layout=Layout();original_reader=layout.get_instance_pose
    m.install(SimpleNamespace(scene_manager=SimpleNamespace(layout_manager=layout)))
    assert layout.check_layout_stability(None)==(False,[0])
    assert layout.get_instance_pose==original_reader
    with pytest.raises(RuntimeError,match='native failure'):layout.check_layout_stability('raise')
    assert layout.get_instance_pose==original_reader


def test_degenerate_convex_geometry_returns_actionable_private_safe_failure():
    from sim.envs.robodojo.scene_geometry import collision_envelope
    from sim.envs.robodojo.public_errors import public_error_message
    points=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],dtype=float)
    with pytest.raises(ValueError,match='requires native cooked geometry') as exc:
        collision_envelope(points,[],'convexHull')
    assert 'requires native cooked geometry' in public_error_message(exc.value)
    assert np.all(points[:,2]==0)  # No jitter, fake thickness, or asset mutation.
