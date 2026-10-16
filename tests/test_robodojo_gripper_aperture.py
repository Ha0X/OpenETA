import numpy as np
import pytest
from sim.envs.robodojo.gripper_aperture import section_bounds, measured_aperture


def calibration():
    return [{'joint':'joint7','sign':1,'limits_m':[0.,.044],'inner_at_zero_m':.0004},
            {'joint':'joint8','sign':-1,'limits_m':[0.,.044],'inner_at_zero_m':-.0004}]


def test_section_interpolates_edges_not_vertex_bounds():
    # Plane z=0 cuts at x=2 and x=3; full mesh bounding box spans x=0..4.
    assert np.allclose(section_bounds([[0,0,-1],[4,0,1],[2,1,-1]],[[0,1,2]]),[2,3])


def test_coplanar_faces_and_missing_section():
    assert section_bounds([[1,0,0],[2,1,0],[3,0,0]],[[0,1,2]]) == [1,3]
    with pytest.raises(ValueError,match='cross'):
        section_bounds([[1,0,1],[2,1,1],[3,0,1]],[[0,1,2]])


def test_both_fingers_and_fixed_offsets_affect_width_and_center():
    a=measured_aperture(calibration(),{'joint7':.01,'joint8':.02})
    assert a['width_m']==pytest.approx(.0308)
    assert a['inner_edges_tcp_x_m']==pytest.approx([-.0204,.0104])
    assert a['width_limits_m']==pytest.approx([.0008,.0888])
    assert measured_aperture(calibration(),{'joint7':0,'joint8':0})['width_m']==pytest.approx(.0008)
    assert measured_aperture(calibration(),{'joint7':.044,'joint8':.044})['width_m']==pytest.approx(.0888)


@pytest.mark.parametrize('q',[float('nan'),float('inf'),-.02,.1])
def test_invalid_reading_does_not_become_plausible_width(q):
    with pytest.raises(ValueError):measured_aperture(calibration(),{'joint7':q,'joint8':.02})


def test_missing_second_finger_is_not_inferred_from_first():
    with pytest.raises(KeyError):measured_aperture(calibration(),{'joint7':.01})


def test_wire_preserves_robot_only_aperture():
    from adapter.protocol import RobotState
    aperture=measured_aperture(calibration(),{'joint7':.01,'joint8':.02})
    state=RobotState.from_dict({'gripper_state':{'aperture':aperture}})
    assert state.to_dict()['gripper_state']['aperture']==aperture
