import json
from copy import deepcopy
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim.envs.robodojo.fk_diagnostics import fk_mismatch_report
from sim.envs.robodojo.native_scene import NativeScene
from sim.envs.robodojo.scene_geometry import forward_kinematics


def model():
    eye = np.eye(4).tolist()
    return {'role': 'left', 'control_owner': 'target', 'base_link': 'base',
            'joint_names': ['finger'], 'body_frames': {'base': eye, 'tip': eye},
            'joints': [{'name': 'finger', 'kind': 'prismatic', 'parent': 'base',
                        'child': 'tip', 'axis': 'Y', 'parent_frame': eye, 'child_frame': eye}]}


def test_finger_axial_residual_does_not_change_model_or_state():
    robot = model(); before = deepcopy(robot); q = np.array([.04])
    measured = forward_kinematics(robot, [.040375])
    report = fk_mismatch_report(robot, q, measured)
    joint = report['local_joint_residuals'][0]
    assert joint['axial_coordinate_error_m'] == pytest.approx(.000375)
    assert joint['transverse_error_m'] == 0
    assert report['base']['matches_existing_guard']
    assert not joint['matches_existing_guard']
    assert robot == before and q.tolist() == [.04]
    json.dumps(report, allow_nan=False)


def test_shared_base_shift_is_not_misdiagnosed_as_finger_overtravel():
    robot = model(); measured = forward_kinematics(robot, [.04])
    for frame in measured.values():
        frame[0, 3] += .002
    report = fk_mismatch_report(robot, [.04], measured)
    assert not report['base']['matches_existing_guard']
    assert len(report['mismatched_links']) == 2
    assert report['local_joint_residuals'][0]['matches_existing_guard']


@pytest.mark.parametrize('kind', ['transverse', 'rotation', 'nan'])
def test_non_axial_error_is_visible_and_json_safe(kind):
    robot = model(); measured = forward_kinematics(robot, [.04])
    if kind == 'transverse': measured['tip'][0, 3] += .001
    if kind == 'rotation': measured['tip'][:3, :3] = np.diag([-1., -1., 1.])
    if kind == 'nan': measured['tip'][0, 3] = np.nan
    report = fk_mismatch_report(robot, [.04], measured)
    assert report['mismatched_links']
    joint = report['local_joint_residuals'][0]
    if kind == 'transverse': assert joint['transverse_error_m'] == pytest.approx(.001)
    if kind == 'rotation': assert joint['rotation_matrix_max_error'] == 2
    if kind == 'nan': assert not joint['finite']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('broken_reporter', [False, True])
def test_native_failure_remains_closed_even_if_diagnostic_fails(monkeypatch, caplog, broken_reporter):
    robot = model()
    scene = NativeScene.__new__(NativeScene)
    scene.world = {}
    scene.scene = S(snapshot={'robots': [robot]})
    entity = S(body_names=['base', 'tip'], data=S(body_link_pose_w=np.array([
        [[0, 0, 0, 1, 0, 0, 0], [0, .041, 0, 1, 0, 0, 0]]])))
    scene.env = S(robot_manager=S(robot_key=[entity]))
    scene.positions = lambda: [np.array([.04])]
    if broken_reporter:
        def fail(*args): raise RuntimeError('diagnostic failed')
        monkeypatch.setattr('sim.envs.robodojo.fk_diagnostics.fk_mismatch_report', fail)
    with pytest.raises(ValueError, match='Runtime physical FK/base differs from collision model'):
        scene.refresh()
    assert 'OPENETA_FK_MISMATCH' in caplog.text
    assert ('OPENETA_FK_DIAGNOSTIC_UNAVAILABLE' if broken_reporter else 'OPENETA_FK_DIAGNOSTIC {') in caplog.text


@pytest.mark.parametrize('base_shift', [0., .001])
def test_native_refresh_uses_actual_finger_without_hiding_base_drift(base_shift):
    robot=model();robot.update(finger_links=['tip'],arm_joint_names=[])
    scene=NativeScene.__new__(NativeScene);scene.world={}
    scene.scene=S(snapshot={'robots':[robot]})
    raw=np.array([[.04]])
    poses=np.array([[[base_shift,0,0,1,0,0,0], [base_shift,.041,0,1,0,0,0]]])
    entity=S(body_names=['base','tip'],data=S(joint_pos=raw,body_link_pose_w=poses))
    scene.env=S(robot_manager=S(robot_key=[entity]))
    if base_shift:
        with pytest.raises(ValueError,match='Runtime physical FK/base differs'):
            scene.refresh()
    else:
        scene.refresh()
        assert scene.positions()[0][0]==pytest.approx(.041)
    assert raw[0,0]==.04
