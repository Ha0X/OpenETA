from types import SimpleNamespace as S
import pytest
from sim.envs.robodojo.cloth_material_profile import PROFILE, ENV, install_factory, inspect_binding


def fixture(monkeypatch):
    monkeypatch.setenv(ENV,PROFILE)
    class Material:
        def __init__(self,**kwargs):self.args=kwargs;self.value=kwargs['friction']
        def get_friction(self):return self.value
    module=S(ParticleMaterial=Material);install_factory(module);install_factory(module)
    return module


def test_only_friction_changes_and_binding_reads_actual_value(monkeypatch):
    m=fixture(monkeypatch);args={'friction':10.,'adhesion':.1,'particle_friction_scale':.5}
    mat=m.ParticleMaterial(**args)
    assert mat.args==dict(args,friction=20.) and args['friction']==10.
    env=S(scene_manager=S(_garment_objects=[{'g':S(particle_material=mat)}]))
    value=inspect_binding(env,PROFILE)
    assert value['materials'][0]['measured_friction']==20.
    assert not value['comparable_to_native_physics_baseline']
    with pytest.raises(ValueError,match='differs'):inspect_binding(env,None)
    mat.value=10.
    with pytest.raises(ValueError,match='changed'):inspect_binding(env,PROFILE)


def test_native_profile_needs_no_material_access(monkeypatch):
    monkeypatch.delenv(ENV,raising=False)
    assert inspect_binding(S(),None) is None
    with pytest.raises(ValueError,match='differs'):inspect_binding(S(),PROFILE)


@pytest.mark.parametrize('bad',[0,-1,float('nan'),float('inf')])
def test_invalid_friction_rejected(monkeypatch,bad):
    m=fixture(monkeypatch)
    with pytest.raises(ValueError):m.ParticleMaterial(friction=bad)
