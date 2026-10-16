import json

from test_codex_robodojo_mark_preview import host, set_frame


def test_mark_omits_only_unchanged_previously_delivered_fields_and_keeps_full_evidence(tmp_path):
    h, remote = host(tmp_path)
    try:
        set_frame(remote, [320,240])
        initial = json.loads(h.call('episode_status', {})[0].text)
        marked = json.loads(h.call('mark_point', {'camera':'head','pixel':[320,240]})[0].text)
        omitted = marked['state_delivery']
        assert omitted['fields'] == ['robot','collision_policy','layout_contract']
        assert omitted['reference'] == {'call_id':'000001','action_index':0}
        assert not set(omitted['fields']) & marked.keys()
        for key in ('action_index','tcp_state','active_support_holds','cameras'):
            assert marked[key] == initial[key]
        assert marked['status']['tool_calls'] == 2
        assert marked['receipt']['mark_preview']['status'] == 'available'
        packet = json.loads((h.output/'calls/000002/after-observation.json').read_text())
        assert packet['observation']['robot'] == remote.packet['observation']['robot']
        full_again = json.loads(h.call('observe', {})[0].text)
        assert 'state_delivery' not in full_again and full_again['robot'] == initial['robot']
    finally:
        h.close('test_complete')


def test_first_mark_and_changed_robot_state_are_never_silently_omitted(tmp_path):
    h, remote = host(tmp_path)
    try:
        set_frame(remote, [320,240])
        first = json.loads(h.call('mark_point', {'camera':'head','pixel':[320,240]})[0].text)
        assert 'robot' in first and 'state_delivery' not in first
        remote.packet['observation']['robot']['joint_positions'] = [.1]*6
        changed = json.loads(h.call('mark_point', {'camera':'head','pixel':[320,240]})[0].text)
        assert changed['robot']['joint_positions'] == [.1]*6 and 'state_delivery' not in changed
        same = json.loads(h.call('mark_point', {'camera':'head','pixel':[320,240]})[0].text)
        assert same['state_delivery']['reference']['call_id'] == '000002'
        error = json.loads(h.call('mark_point', {'camera':'head','pixel':[1.5,0]})[0].text)
        assert error['error'] and 'robot' in error and 'state_delivery' not in error
    finally:
        h.close('test_complete')
