import sys
from types import SimpleNamespace as NS
import numpy as np
import pytest

from sim.envs.robodojo.native_cloth import NativeClothReader
from sim.envs.robodojo.deformable_surface import DeformableCoverageError


def native(monkeypatch, device='cpu', valid=False):
    class Attribute:
        def __init__(self,value): self.value=value
        def Get(self): return self.value
    points=np.array([[0.,0.,0.],[.01,0,0],[0,.01,0]])
    indices=Attribute([0,1,2]);counts=Attribute([3])
    point_attr=Attribute(points)
    mesh=NS(GetFaceVertexCountsAttr=lambda:counts,GetFaceVertexIndicesAttr=lambda:indices,GetPointsAttr=lambda:point_attr)
    attrs={key:Attribute(value) for key,value in {'particleSystemEnabled':True,'nonParticleCollisionEnabled':True,'contactOffset':.008}.items()}
    g=NS(_device=device,prim=object(),particle_system=NS(prim=NS(GetAttribute=lambda key:attrs[key])),
         _cloth_prim_view=NS(is_physics_handle_valid=lambda:valid),sample_mesh_vertices=lambda:(points.copy(),None,None,None))
    monkeypatch.setitem(sys.modules,'pxr',NS(UsdGeom=NS(Mesh=lambda _:mesh,Xformable=lambda _:NS(ComputeLocalToWorldTransform=lambda _:np.eye(4))),Usd=NS(TimeCode=NS(Default=lambda:None))))
    return g,points,indices,attrs


def test_cpu_read_uses_world_deformed_vertices_without_root_pose_fallback(monkeypatch):
    g,p,_,_=native(monkeypatch)
    r=NativeClothReader(g,use_fabric=False,simulation_time_s=0)
    p[:,2]=.04
    frame=r.refresh(simulation_time_s=.004)
    assert np.all(frame.vertices[:,2]==.04)
    assert r.source=='native_cpu_usd_physics_writeback'


def test_cpu_fabric_without_physics_writeback_is_rejected(monkeypatch):
    g,_,_,_=native(monkeypatch)
    with pytest.raises(DeformableCoverageError,match='writeback'):
        NativeClothReader(g,use_fabric=True,simulation_time_s=0)


def test_gpu_requires_valid_live_particle_handle(monkeypatch):
    g,_,_,_=native(monkeypatch,device='cuda:0')
    with pytest.raises(DeformableCoverageError,match='live particle'):
        NativeClothReader(g,use_fabric=True,simulation_time_s=0)


@pytest.mark.parametrize('attribute',['particleSystemEnabled','nonParticleCollisionEnabled'])
def test_disabled_native_cloth_collision_is_not_silently_supported(monkeypatch,attribute):
    g,_,_,attrs=native(monkeypatch);attrs[attribute].value=False
    with pytest.raises(DeformableCoverageError,match='not enabled'):
        NativeClothReader(g,use_fabric=False,simulation_time_s=0)


def test_topology_replacement_invalidates_old_contact_references(monkeypatch):
    g,_,indices,_=native(monkeypatch)
    r=NativeClothReader(g,use_fabric=False,simulation_time_s=0)
    indices.value=[2,1,0]
    with pytest.raises(DeformableCoverageError,match='topology changed'):
        r.refresh(simulation_time_s=.004)


def test_failed_live_read_does_not_fall_back_to_static_initial_mesh(monkeypatch):
    g,_,_,_=native(monkeypatch)
    r=NativeClothReader(g,use_fabric=False,simulation_time_s=0)
    def fail():raise RuntimeError('physics handle unavailable')
    r.points_attribute.Get=fail
    with pytest.raises(RuntimeError,match='unavailable'):r.refresh(simulation_time_s=.004)


def test_direct_cpu_read_must_agree_with_native_world_coordinates(monkeypatch):
    g,points,_,_=native(monkeypatch)
    g.sample_mesh_vertices=lambda:(points+[.1,0,0],None,None,None)
    with pytest.raises(DeformableCoverageError,match='differs'):
        NativeClothReader(g,use_fabric=False,simulation_time_s=0)


def test_runtime_transform_is_fresh_not_cached_at_authorization(monkeypatch):
    g,points,_,_=native(monkeypatch)
    r=NativeClothReader(g,use_fabric=False,simulation_time_s=0)
    transform=np.eye(4);transform[:3,:3]=np.diag([2,3,4]);transform[2,3]=.07
    r.xform.ComputeLocalToWorldTransform=lambda _:transform.T
    frame=r.refresh(simulation_time_s=.004)
    assert np.allclose(frame.vertices,points@[ [2,0,0],[0,3,0],[0,0,4] ]+[0,0,.07])
