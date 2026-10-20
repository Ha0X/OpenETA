import numpy as np
import pytest
from sim.envs.robodojo.native_fluid import physical_points, FluidCoverageError


def test_smoothing_uses_physical_positions_not_rendered_liquid():
    rendered=[[0,0,0],[1,0,0]];physical=[[0,0,.01],[1,0,.02]]
    points,source=physical_points(rendered,physical,smoothing_enabled=True)
    assert np.array_equal(points,physical)
    assert source=='physx_particle_simulation_points'


@pytest.mark.parametrize('physical',[None,[],[[0,0,0]],[[0,0,0],[1,float('nan'),0]]])
def test_smoothing_never_falls_back_to_render_positions(physical):
    with pytest.raises(FluidCoverageError):
        physical_points([[0,0,0],[1,0,0]],physical,smoothing_enabled=True)


def test_unsmoothed_physics_does_not_use_stale_simulation_points():
    original=np.array([[0.,0.,0.]])
    points,source=physical_points(original,[[1,2,3]],smoothing_enabled=False)
    original[0,0]=3
    assert points[0,0]==0 and source=='unsmoothed_physx_usd_positions'


@pytest.mark.parametrize('render',[None,[],[1,2,3],[[0,0,float('inf')]]])
def test_invalid_directory_is_rejected(render):
    with pytest.raises(FluidCoverageError):physical_points(render,None,smoothing_enabled=False)


def test_unknown_smoothing_is_rejected():
    with pytest.raises(FluidCoverageError):physical_points([[0,0,0]],None,smoothing_enabled=None)
