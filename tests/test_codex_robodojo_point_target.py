import json

import numpy as np
import pytest

from test_codex_robodojo_host import host
from tools.codex_robodojo_host import target_matrix


def result(blocks):
    return json.loads(blocks[0].text)


def test_offset_point_reaches_requested_world_position_after_rotation():
    target={'xyz':[.3,-.1,.8], 'point_in_tcp':[.12,-.03,.04],
            'approach':[1,0,0], 'jaw':[0,1,0]}
    matrix=np.asarray(target_matrix(target,np.eye(4)))
    assert np.allclose(matrix[:3,:3] @ target['point_in_tcp'] + matrix[:3,3], target['xyz'])
    plain={k:v for k,v in target.items() if k!='point_in_tcp'}
    assert np.allclose(np.asarray(target_matrix(plain,np.eye(4)))[:3,3],target['xyz'])


def test_public_mark_coordinate_conversion_uses_selected_robot_pose_only(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        tcp=np.array(r.packet['control']['tcp_poses']['right'])
        local=np.array([.12,-.02,.04]);world=tcp[:3,:3]@local+tcp[:3,3]
        mark={'mark_id':'visible_point','camera':'head','pixel':[30,40],
              'xyz':world.tolist(),'source_action_index':0}
        r.packet['receipt']={'mark':mark}
        reply=result(h.call('mark_point',{'camera':'head','pixel':[30,40],'relative_to_tcp':'right'}))
        assert reply['error'] is None
        assert reply['receipt']['mark']==mark
        assert np.allclose(reply['receipt']['tcp_point']['point_in_tcp'],local)
        assert reply['receipt']['tcp_point']['role']=='right'
        assert not reply['receipt']['tcp_point']['object_identity_or_attachment_inferred']
        assert 'relative_to_tcp' not in r.calls[-1][1]['request']
    finally:h.close('test_complete')


def test_waypoint_offsets_are_explicit_and_conversion_is_reported(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        request={'role':'right','waypoints':[{'xyz':[.2,0,.9]}],
                 'target':{'xyz':[.3,0,.8],'point_in_tcp':[.1,0,0]}}
        reply=result(h.call('move_to',request))
        assert reply['error'] is None
        targets=np.array(r.calls[-1][1]['request']['targets']['right'])
        assert np.allclose(targets[0,:3,3],[.2,0,.9])
        assert np.allclose(targets[1,:3,3]+targets[1,:3,:3]@[.1,0,0],[.3,0,.8])
        conversion=reply['receipt']['host_adjustments'][-1]
        assert conversion['role']=='right' and conversion['waypoint_index']==1
        assert np.allclose(conversion['effective_tcp_xyz'],targets[1,:3,3])
        assert not conversion['actual_object_point_measured_after_motion']
    finally:h.close('test_complete')


@pytest.mark.parametrize('point', [[0,0], [0,0,0,0], [float('nan'),0,0], [0,float('inf'),0]])
def test_bad_offsets_reject_before_physics_dispatch(tmp_path,point):
    h,r=host(tmp_path)
    try:
        before=len(r.calls)
        reply=result(h.call('move_to',{'role':'primary','target':{'xyz':[0,0,.9],'point_in_tcp':point}}))
        assert reply['error']['code']=='request_rejected' and len(r.calls)==before
    finally:h.close('test_complete')


def test_point_target_does_not_add_contact_authorization(tmp_path):
    h,r=host(tmp_path)
    try:
        h.call('move_to',{'role':'primary','target':{'xyz':[0,0,.9],'point_in_tcp':[0,0,.2]}})
        request=r.calls[-1][1]['request']
        assert not set(request)&{'contact_mark','contact_marks','tool_contact','support_contact'}
    finally:h.close('test_complete')


def test_failed_motion_still_reports_coordinate_conversion(tmp_path):
    h,r=host(tmp_path)
    try:
        r.packet.update(admitted=False,receipt={'status':'rejected','native_actions':0,'physics_steps':0})
        reply=result(h.call('move_to',{'role':'primary','target':{'xyz':[0,0,.9],'point_in_tcp':[0,0,.2]}}))
        assert reply['error']['code']=='motion_not_admitted'
        assert reply['receipt']['status']=='rejected'
        assert reply['receipt']['host_adjustments'][-1]['kind']=='explicit_tcp_point_target_conversion'
    finally:h.close('test_complete')
