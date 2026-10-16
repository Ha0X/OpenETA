import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.codex_robodojo_host import RoboDojoHost, target_matrix
from tools.harness_config import ConfigRegistry
from sim.envs.robodojo.outcome import SCHEMA, UPSTREAM_REVISION
from sim.envs.robodojo.x5_calibration import TCP_FROM_LINK
from sim.envs.robodojo.layout import bind_layout, describe_layout


def layout(dual=False):
    native = describe_layout({'robots':[{'robot_name':'x5'}]*2})
    return bind_layout(native, {'robots':[{'robot_name':'x5'}]*(2 if dual else 1)},
                       'dual_x5' if dual else 'single_x5')


def packet():
    return {'admitted':True,'action_index':0,'receipt':None,
        'control':{'controller_id':'robodojo.checked_joint_ik.v1','roles':['primary'],
                   'layout_contract':layout(),
                   'tcp_poses':{'primary':TCP_FROM_LINK.tolist()},'collision_policy':{},
                   'source_sha256':ConfigRegistry().resolve('robodojo_single_x5_left_mount_v1').value['components']['interface']['source_sha256']},
        'observation':{'task':'Pick up the scissors','objects':[],'robot':{'joint_names':[f'joint{i}' for i in range(1,7)],
            'end_effector_pose':{'xyz':[0,0,0],'quat_xyzw':[0,0,0,1]},
            'metadata':{'robot_contract':'openeta.x5_parallel_link6_tcp.v1'}},
            'cameras':[{'frame_id':'head','width':640,'height':480,
                'intrinsics':{'fx':200,'fy':200,'cx':320,'cy':240},
                'extrinsics':{'camera_to_world':np.eye(4).tolist(),'camera_frame':'opencv',
                              'reference_frame':'world','depth_convention':'optical_z_m'}}]},
        'info':{'official_outcome':{'schema_version':SCHEMA,'source':'robodojo.eval_env.terminal_flags',
            'source_contract_revision':UPSTREAM_REVISION,'terminal':False,'native_execution_ok':True,
            'reward_registered':True,'action_index':0,'status':'running','benchmark_success':None}}}


class Remote:
    def __init__(self):
        self.packet=packet()
        self.calls=[]
        self.fail_move=False
    def call_tool(self, name, args, **kwargs):
        self.calls.append((name,copy.deepcopy(args)))
        if name=='create_env':return {'session_id':'owned','handle':'h'}
        if name=='reset_env':return {}
        if name=='close_env':return {'ok':True}
        if self.fail_move and args['request']['operation']=='move':raise ConnectionError('reply lost after dispatch')
        return copy.deepcopy(self.packet)


def host(tmp_path, dual=False, cloth=False, physical=False, visual_cloth=False, friction=False):
    config=ConfigRegistry().resolve('robodojo_dual_x5_checked_v1' if dual else 'robodojo_single_x5_left_mount_v1')
    args=SimpleNamespace(output=tmp_path/'host',sim_url='http://example.invalid',
        env_id='openeta/robodojo_single_x5_general_pickup-v0',task='Pick up the scissors',seed=0,
        timeout=300,max_requests=20)
    if dual:
        args.env_id='openeta/robodojo_dual_x5_general_pickup-v0'
    binding=config.bind_task(env_id=args.env_id,task=args.task,seed=args.seed)
    remote=Remote()
    if dual:
        args.env_id='openeta/robodojo_dual_x5_general_pickup-v0'
        binding=config.bind_task(env_id=args.env_id,task=args.task,seed=args.seed)
        p=remote.packet
        root=p['observation']['robot']
        right=copy.deepcopy(root);right['end_effector_pose']['xyz']=[.6,0,0]
        p['observation']['robot']={'arms':{'left':root,'right':right}}
        right_tcp=TCP_FROM_LINK.copy();right_tcp[0,3]+=.6
        p['control'].update(roles=['left','right'],tcp_poses={'left':TCP_FROM_LINK.tolist(),'right':right_tcp.tolist()},
                           layout_contract=layout(dual=True),
                           articulations={role:{'id':role} for role in ['left','right']})
    if cloth:
        assert dual
        config=ConfigRegistry().resolve('robodojo_dual_x5_cloth_v1')
        args.env_id='openeta/robodojo_dual_x5_fold_clothes-v0'
        binding=config.bind_task(env_id=args.env_id,task=args.task,seed=args.seed)
        remote.packet['control']['collision_policy']['deformables']='cloth_short_horizon_v1'
    if physical:
        assert not cloth
        config=ConfigRegistry().resolve('robodojo_dual_x5_physical_contact_v1' if dual else 'robodojo_single_x5_physical_contact_v1')
        binding=config.bind_task(env_id=args.env_id,task=args.task,seed=args.seed)
        remote.packet['control']['collision_policy']['contact_policy']='physical_contact_v1'
    if visual_cloth:
        assert dual
        suffix='_friction_2x' if friction else ''
        config=ConfigRegistry().resolve('robodojo_dual_x5_visual_cloth'+suffix+'_v74')
        args.env_id='openeta/robodojo_dual_x5_fold_clothes-v0'
        binding=config.bind_task(env_id=args.env_id,task=args.task,seed=args.seed)
        remote.packet['control']['collision_policy'].update(deformables='cloth_visual_short_horizon_v2',contact_policy='physical_contact_v1')
        if friction:
            remote.packet['control']['physics_profile']={'profile':'cloth_friction_2x_v1',
                'comparable_to_native_physics_baseline':False,
                'materials':[{'native_constructor_friction':10.,'requested_friction':20.,'measured_friction':20.,'multiplier':2.}]}
    return RoboDojoHost(args,config,binding,transport=remote),remote


def test_cloth_contact_mode_requires_explicit_mark_and_is_forwarded_per_arm(tmp_path):
    h,r=host(tmp_path,dual=True,cloth=True)
    try:
        result=json.loads(h.call('move_to',{'role':'left','cloth_contact':'local_layers',
                                          'target':{'xyz':[0,0,1]}})[0].text)
        assert result['error']['code']=='request_rejected'
        h.call('move_to',{'motions':[{'role':'left','cloth_contact':'local_layers','contact_mark':'fresh',
                                    'target':{'xyz':[0,0,1]}}]})
        request=r.calls[-1][1]['request']
        assert request['cloth_contacts']=={'left':'local_layers'}
        assert request['contact_marks']=={'left':'fresh'}
    finally:h.close('test')


def test_rigid_profile_does_not_expose_cloth_contact_semantics(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        result=json.loads(h.call('move_to',{'role':'left','cloth_contact':'local_layers',
            'contact_mark':'fresh','target':{'xyz':[0,0,1]}})[0].text)
        assert result['error']['code']=='request_rejected'
    finally:h.close('test')


def test_tcp_axes_preserved_or_explicit_and_not_quaternion_guessed():
    current=target_matrix({'xyz':[1,2,3],'approach':[0,0,-1],'jaw':[0,1,0]},np.eye(4))
    changed=np.array(target_matrix({'xyz':[2,3,4]},current))
    assert np.allclose(changed[:3,2],[0,0,-1])
    assert np.allclose(changed[:3,0],[0,1,0])
    assert np.isclose(np.linalg.det(changed[:3,:3]),1)
    with pytest.raises(ValueError,match='both'):target_matrix({'xyz':[0,0,0],'approach':[0,0,-1]},current)
    with pytest.raises(ValueError,match='perpendicular'):target_matrix({'xyz':[0,0,0],'approach':[0,0,1],'jaw':[0,0,1]},current)


def test_mcp_termination_during_reset_closes_owned_environment(tmp_path, monkeypatch):
    from tools import codex_mcp_server as server
    handlers = {}
    previous = object()
    monkeypatch.setattr(server.signal, 'getsignal', lambda sig: previous)
    monkeypatch.setattr(server.signal, 'signal', lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(server.sys, 'argv', ['mcp-server'])
    monkeypatch.setenv('OPENETA_CODEX_HOST_ARGS', '[]')
    monkeypatch.setattr(server, 'parser', lambda: SimpleNamespace(
        parse_args=lambda argv: SimpleNamespace(expected_controller='robodojo_checked')))
    original_call = Remote.call_tool
    closes = []
    def interrupted_call(self, name, args, **kwargs):
        if name == 'reset_env':
            handlers[server.signal.SIGTERM](server.signal.SIGTERM, None)
        if name == 'close_env':
            closes.append(dict(args))
        return original_call(self, name, args, **kwargs)
    monkeypatch.setattr(Remote, 'call_tool', interrupted_call)
    monkeypatch.setattr(server, 'build_host', lambda args: host(tmp_path)[0])
    server.main()
    assert closes == [{'session_id': 'owned', 'handle': 'h'}]
    assert handlers[server.signal.SIGTERM] is previous
    status = json.loads((tmp_path / 'host/host-status.json').read_text())
    assert status['closed'] and status['cleanup']['ok']


def test_support_contact_is_explicit_and_batch_placement_rejects_before_dispatch(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        owned=json.loads((tmp_path/'host/owned-environment.json').read_text())
        assert owned['session_id']=='owned' and owned['handle']=='h'
        request={'role':'left','target':{'xyz':[0,0,.9]},'support_contact':{'mode':'place','surface_mark':'mark_table'}}
        h.call('move_to',request)
        assert r.calls[-1][1]['request']['support_contact']==request['support_contact']
        count=len(r.calls)
        response=json.loads(h.call('move_to',{'motions':[request,{'role':'right','target':{'xyz':[.6,0,.9]}}]})[0].text)
        assert response['error']['code']=='request_rejected' and len(r.calls)==count
    finally:h.close('test_complete')


def test_false_success_claim_keeps_episode_open_and_official_outcome(tmp_path):
    h,r=host(tmp_path)
    result=json.loads(h.call('finish_episode',{'success':True,'reason':'I closed the gripper'})[0].text)
    assert result['error']['code']=='request_rejected'
    assert not result['status']['official_task_success'] and not h.closed
    h.close()
    assert h.cleanup['ok']


def test_table_only_slide_forwards_explicit_empty_workpiece_marks(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        request={'role':'left','target':{'xyz':[0,0,.9]},'tool_contact':{
            'mode':'stroke','surface_mark':'table','target_marks':[]}}
        h.call('move_to',request)
        assert r.calls[-1][1]['request']['tool_contact']==request['tool_contact']
    finally:h.close('test_complete')


def test_tool_stroke_contract_is_forwarded_and_incompatible_requests_never_dispatch(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        request={'role':'right','target':{'xyz':[.1,0,.8]},'tool_contact':{
            'mode':'stroke','surface_mark':'table','target_marks':['block','receiver']}}
        h.call('move_to',request)
        assert r.calls[-1][1]['request']['tool_contact']==request['tool_contact']
        for invalid in [
            {**request,'support_contact':{'mode':'place','surface_mark':'table'}},
            {'motions':[request]},
            {**request,'tool_contact':{**request['tool_contact'],'target_marks':['block']*9}},
            {**request,'tool_contact':{**request['tool_contact'],'target_handles':['workpiece_prior']*7}},
            {**request,'tool_contact':{**request['tool_contact'],'target_handles':['']}},
        ]:
            count=len(r.calls)
            response=json.loads(h.call('move_to',invalid)[0].text)
            assert response['error']['code']=='request_rejected' and len(r.calls)==count
        hold={'role':'left','target':{'xyz':[0,0,.8]},'support_contact':{
            'mode':'place','surface_mark':'table','retain_support':True}}
        h.call('move_to',hold)
        assert r.calls[-1][1]['request']['support_contact']['retain_support'] is True
    finally:h.close('test_complete')


def test_workpiece_references_forward_without_pose_or_implicit_marks(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        request={'role':'right','target':{'xyz':[.1,0,.8]},'tool_contact':{
            'mode':'stroke','surface_mark':'fresh_table','target_marks':[],
            'target_handles':['workpiece_prior']}}
        h.call('move_to',request)
        assert r.calls[-1][1]['request']['tool_contact']==request['tool_contact']
    finally:h.close('test_complete')


def test_motion_preserves_waypoints_and_fresh_state_and_no_unknown_retry(tmp_path):
    h,r=host(tmp_path)
    args={'role':'primary','target':{'xyz':[0,0,.2]},'waypoints':[{'xyz':[0,.1,.1]}]}
    h.call('move_to',args)
    command=r.calls[-1][1]['request']
    assert command['action_index']==0
    assert len(command['targets']['primary'])==2
    assert np.allclose(np.array(command['targets']['primary'][0])[:3,3],[0,.1,.1])
    before=len(r.calls)
    r.fail_move=True
    result=json.loads(h.call('move_to',args)[0].text)
    assert result['error']['code']=='transport_or_execution_outcome_unknown'
    assert [name for name,_ in r.calls[before:]]==['configured_robodojo','close_env']
    assert h.closed


def test_strict_tool_schema_rejects_unavailable_effector_without_dispatch(tmp_path):
    h,r=host(tmp_path)
    before=len(r.calls)
    result=json.loads(h.call('gripper_control',{'role':'left','openness':0})[0].text)
    assert result['error'] and len(r.calls)==before
    h.close()


def test_dual_batch_defaults_to_independent_coordination_and_rejects_duplicate_roles(tmp_path):
    h,r=host(tmp_path,dual=True)
    motions=[{'role':'left','target':{'xyz':[.1,0,.2]}},{'role':'right','target':{'xyz':[.7,0,.2]}}]
    before=len(r.calls)
    reply=json.loads(h.call('move_to',{'motions':motions})[0].text)
    assert reply['error'] is None and len(r.calls)==before+1
    request=r.calls[-1][1]['request']
    assert set(request['targets'])=={'left','right'}
    assert request['coordination'] == 'independent'
    h.call('move_to', {'motions':motions, 'coordination':'synchronized'})
    assert r.calls[-1][1]['request']['coordination'] == 'synchronized'
    before=len(r.calls)
    reply=json.loads(h.call('move_to',{'motions':[motions[0],motions[0]]})[0].text)
    assert reply['error'] and len(r.calls)==before
    h.close()


def test_agent_timestamps_are_not_part_of_motion_schema(tmp_path):
    h, r = host(tmp_path, dual=True)
    before = len(r.calls)
    result = json.loads(h.call('move_to', {'motions':[
        {'role':'left','target':{'xyz':[0,0,.2]},'start_at_s':3},
        {'role':'right','target':{'xyz':[.5,0,.2]}}]})[0].text)
    assert result['error']['code'] == 'request_rejected' and len(r.calls) == before
    h.close()


def test_call_evidence_keeps_historical_rgbd_and_exact_return_selection(tmp_path):
    import base64
    import hashlib
    h, r = host(tmp_path)
    frame = r.packet['observation']['cameras'][0]
    frame.update(frame_id='../../outside', rgb_base64=base64.b64encode(b'first-rgb').decode(),
                 depth_base64=base64.b64encode(b'first-depth').decode())
    h.call('observe', {})
    first = h.output/'calls/000001'
    historical = (first/'after-observation.json').read_bytes()
    frame['rgb_base64'] = base64.b64encode(b'second-rgb').decode()
    result = h.call('observe', {'cameras':[]})
    assert len(result) == 1
    assert (first/'after-observation.json').read_bytes() == historical
    assert (first/'after-000-rgb.png').read_bytes() == b'first-rgb'
    assert (first/'after-000-depth.png').read_bytes() == b'first-depth'
    assert (h.output/'calls/000002/after-000-rgb.png').read_bytes() == b'second-rgb'
    response = json.loads((first/'response.json').read_text())
    assert response['content'][-1]['sha256'] == hashlib.sha256(b'first-rgb').hexdigest()
    assert json.loads(result[0].text)['evidence']['observation'] == 'received_this_call'
    h.close()


def test_unknown_motion_retains_before_evidence_without_inventing_after(tmp_path):
    h, r = host(tmp_path)
    r.fail_move = True
    result = json.loads(h.call('move_to', {'role':'primary','target':{'xyz':[0,0,.2]}})[0].text)
    assert result['evidence']['observation'] == 'outcome_unknown_last_known'
    assert (h.output/'calls/000001/before-observation.json').exists()
    assert not (h.output/'calls/000001/after-observation.json').exists()
    assert (h.output/'calls/000001/response.json').exists()
    assert h.closed


def test_journal_failure_before_action_never_dispatches(tmp_path, monkeypatch):
    import tools.codex_robodojo_host as module
    h, r = host(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(module, 'CallEvidence', fail)
    before = len(r.calls)
    result = json.loads(h.call('gripper_control', {'role':'primary','openness':0})[0].text)
    assert result['error']['action_dispatched'] is False
    assert result['error']['code'] == 'evidence_write_failed_before_dispatch'
    assert len(r.calls) == before and h.stopping
    h.close()


def test_journal_failure_after_action_never_replays(tmp_path, monkeypatch):
    from tools.harness_evidence import CallEvidence
    h, r = host(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(CallEvidence, 'complete', fail)
    before = len(r.calls)
    result = json.loads(h.call('gripper_control', {'role':'primary','openness':0})[0].text)
    assert len(r.calls) == before + 1
    assert result['error']['automatic_retry_allowed'] is False and h.stopping
    h.close()


def test_server_invalid_state_is_unknown_not_an_ordinary_rejection(tmp_path):
    h, r = host(tmp_path)
    original = r.call_tool
    def invalid(name, args, **kwargs):
        if name == 'configured_robodojo':
            return {'error':'configured_operation_rejected', 'state_valid':False,
                    'message':'Execution state is invalid'}
        return original(name, args, **kwargs)
    r.call_tool = invalid
    result = json.loads(h.call('gripper_control', {'role':'primary','openness':0})[0].text)
    assert result['error']['code'] == 'transport_or_execution_outcome_unknown'
    assert result['evidence']['observation'] == 'outcome_unknown_last_known'
    assert h.closed and h.cleanup['ok']


def test_unchanged_preview_images_omitted_only_after_they_were_shown(tmp_path):
    import base64
    h,r=host(tmp_path)
    try:
        frame=r.packet['observation']['cameras'][0]
        frame.update(rgb_base64=base64.b64encode(b'rgb').decode(),depth_base64=base64.b64encode(b'depth').decode())
        r.packet['receipt']={'native_actions':0,'physics_steps':0,'reached_target':False,
                             'stop':{'reason':'ik_search_no_valid_solution'}}
        request={'role':'primary','target':{'xyz':[0,0,.2]}}
        first=h.call('move_to',request)
        assert any(b.type=='image' for b in first)
        second=h.call('move_to',request)
        assert not any(b.type=='image' for b in second)
        assert json.loads(second[0].text)['image_delivery']['reference_action_index']==0
        assert json.loads(second[0].text)['receipt']['stop']==r.packet['receipt']['stop']
        assert any(b.type=='image' for b in h.call('observe',{}))
        frame['depth_base64']=base64.b64encode(b'new-depth').decode()
        assert any(b.type=='image' for b in h.call('move_to',request))
        r.packet['receipt']['native_actions']=1;r.packet['receipt']['physics_steps']=10
        assert any(b.type=='image' for b in h.call('move_to',request))
    finally:h.close('test_complete')


def test_status_exposes_separate_simulation_and_request_budgets(tmp_path):
    h,r=host(tmp_path)
    try:
        r.packet['control']['episode_budget']={'action_limit':1000,'used_actions':790,
            'remaining_actions':210,'terminal':False,'source':'openeta_adapter_max_episode_steps'}
        status=json.loads(h.call('episode_status',{})[0].text)['status']
        assert status['simulation_budget']==r.packet['control']['episode_budget']
        assert status['remaining_requests']==h.args.max_requests-h.tool_calls
    finally:h.close('test_complete')


def test_move_transport_outlasts_worker_budget_without_extending_status(tmp_path,monkeypatch):
    h,remote=host(tmp_path);timeouts=[];original=remote.call_tool
    def call(name,args,**kwargs):
        timeouts.append(kwargs['timeout_s'])
        return original(name,args,**kwargs)
    monkeypatch.setattr(remote,'call_tool',call)
    h._remote({'operation':'move'})
    h._remote({'operation':'status'})
    assert timeouts==[420,240]
    h.close('test')


@pytest.mark.parametrize('mode', ['adjust', 'reorient'])
def test_contact_reorientation_dispatches_explicit_mode_without_pose_repair(tmp_path, mode):
    h, r = host(tmp_path)
    try:
        result = json.loads(h.call('move_to', {'role':'primary',
            'target':{'xyz':[0,0,.2]},
            'tool_contact':{'mode':mode,'surface_mark':'fresh', 'target_marks':[]}})[0].text)
        assert result['error'] is None
        request = r.calls[-1][1]['request']
        assert request['tool_contact']['mode'] == mode
        assert np.allclose(np.asarray(request['targets']['primary'][0])[:3,3], [0,0,.2])
    finally:
        h.close('test')


def test_nonprehensile_push_forwarding_and_exclusive_scope(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        request={'role':'right','target':{'xyz':[.1,0,.8]},'push_contact':{
            'body_mark':'ruler','surface_mark':'table','target_marks':['block'],
            'fingers':['finger_1']}}
        result=json.loads(h.call('move_to',request)[0].text)
        assert result['error'] is None
        assert r.calls[-1][1]['request']['push_contact']==request['push_contact']
        count=len(r.calls)
        for invalid in [
            {**request,'contact_mark':'ruler'},
            {**request,'waypoints':[]},
            {'motions':[request]},
            {**request,'push_contact':{**request['push_contact'],'fingers':['palm']}},
            {**request,'support_contact':{'mode':'place','surface_mark':'table'}}]:
            result=json.loads(h.call('move_to',invalid)[0].text)
            assert result['error']['code']=='request_rejected' and len(r.calls)==count
    finally:h.close('test_complete')


def test_extended_budget_is_operator_bound_and_live_verified(tmp_path):
    h,r=host(tmp_path,dual=True)
    try:
        h.config=ConfigRegistry().resolve('robodojo_dual_x5_align_blocks_400_v1')
        h._remote({'operation':'status'})
        assert r.calls[-1][1]['request']['episode_budget_override']=='align_blocks_400_v1'
        with pytest.raises(ValueError,match='episode budget'):h._validate_binding()
        protocol={'override_id':'align_blocks_400_v1','original_native_action_limit':200,
                  'effective_native_action_limit':400,'comparable_to_native_budget_baseline':False}
        h.packet['control']['episode_budget']={'protocol':protocol}
        h._validate_binding()
        protocol['effective_native_action_limit']=200
        with pytest.raises(ValueError,match='episode budget'):h._validate_binding()
        assert all('episode_budget_override' not in str(t.inputSchema) for t in h.schemas.values())
    finally:h.close('test')


def test_generic_contact_profile_accepts_unlabelled_rotation_waypoints_and_dual_motion(tmp_path):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        schema=h.schemas['move_to'].inputSchema['oneOf'][0]['properties']
        assert not set(schema)&{'push_contact','support_contact','tool_contact','contact_mark'}
        target={'xyz':[.1,0,.5],'approach':[0,0,-1],'jaw':[1,0,0]}
        args={'motions':[{'role':role,'target':target,'waypoints':[{'xyz':[0,0,.6]}]} for role in ['left','right']]}
        result=json.loads(h.call('move_to',args)[0].text)
        assert not result.get('error')
        request=r.calls[-1][1]['request']
        assert request['contact_policy']=='physical_contact_v1'
        assert set(request['targets'])=={'left','right'}
        assert len(request['targets']['left'])==2
        result=json.loads(h.call('gripper_control',{'role':'right','openness':.4})[0].text)
        assert not result.get('error')
        guide=json.loads(h.call('read_skill',{'skill_id':'push-slide'})[0].text)['receipt']
        refs=[d['reference'] for d in guide['documents']]
        assert refs==['references/push-slide.md','references/contracts/physical-contact.md']
        r.packet['control']['collision_policy'].pop('contact_policy')
        h.packet=r.packet
        with pytest.raises(ValueError,match='contact policy'):h._validate_binding()
    finally:h.close('test')


def test_physical_relative_target_and_wait_are_explicit(tmp_path):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        h.call('move_to',{'role':'left','target':{'delta_xyz':[0,0,.001],'frame':'tcp'}})
        request=r.calls[-1][1]['request']
        old=np.asarray(r.packet['control']['tcp_poses']['left'])
        new=np.asarray(request['targets']['left'][0])
        assert np.allclose(new[:3,3],old[:3,3]+old[:3,2]*.001)
        result=json.loads(h.call('wait',{'duration_s':.1})[0].text)
        assert not result.get('error')
        assert r.calls[-1][1]['request']['operation']=='wait'
        assert r.calls[-1][1]['request']['duration_s']==.1
        before=len(r.calls)
        result=json.loads(h.call('wait',{'duration_s':3.})[0].text)
        assert result['error']['code']=='request_rejected' and len(r.calls)==before
    finally:h.close('test')


def test_legacy_profiles_reject_new_relative_and_wait(tmp_path):
    h,r=host(tmp_path)
    try:
        before=len(r.calls)
        for name,args in [('wait',{'duration_s':.1}),('move_to',{'role':'primary','target':{'delta_xyz':[0,0,.001]}})]:
            result=json.loads(h.call(name,args)[0].text)
            assert result['error']['code']=='request_rejected'
        assert len(r.calls)==before
    finally:h.close('test')


@pytest.mark.parametrize('value',[
    {'xyz':[0,0,0],'delta_xyz':[0,0,.01]}, {'delta_xyz':[0,0,.051]},
    {'delta_xyz':[0,0,.01],'point_in_tcp':[0,0,0]}, {'delta_xyz':[0,0,.01],'frame':'hidden'},
    {'xyz':[0,0,0],'frame':'tcp'}, {'delta_xyz':[0,0,.01],'approach':[0,0,1],'jaw':[1,0,0]}])
def test_relative_translation_rejects_ambiguous_or_unbounded_input(value):
    with pytest.raises(ValueError):target_matrix(value,np.eye(4))


def test_physical_precision_forwarding_and_batch_scope(tmp_path):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        motion={'role':'left','target':{'delta_xyz':[0,0,.001]}}
        for args in [{**motion,'position_tolerance_m':.0005},
                     {'motions':[motion],'position_tolerance_m':.001}]:
            payload=json.loads(h.call('move_to',args)[0].text)
            assert payload.get('error') is None
            assert r.calls[-1][1]['request']['position_tolerance_m']==args['position_tolerance_m']
        before=len(r.calls)
        invalid=[{**motion,'position_tolerance_m':.0001},
                 {'motions':[{**motion,'position_tolerance_m':.001}]}]
        for args in invalid:
            payload=json.loads(h.call('move_to',args)[0].text)
            assert payload['error']['code']=='request_rejected'
        assert len(r.calls)==before
    finally:h.close('test')


def test_gpu_profile_requires_measured_pipeline_and_is_forwarded(tmp_path):
    h,remote=host(tmp_path,dual=True,physical=True)
    try:
        h.config=ConfigRegistry().resolve('robodojo_dual_x5_gpu_v73')
        with pytest.raises(ValueError,match='GPU simulation runtime'):
            h._validate_binding()
        runtime={'profile':'gpu_tensor_v1','physics_tensor_device':'cuda:0','use_fabric':True}
        h.packet['control']['simulation_runtime']=runtime
        h.packet['control']['source_sha256']=h.config.value['components']['interface']['source_sha256']
        h._validate_binding()
        remote.packet=copy.deepcopy(h.packet)
        h._remote({'operation':'status'})
        assert remote.calls[-1][1]['request']['simulation_profile']=='gpu_tensor_v1'
        h.config=ConfigRegistry().resolve('robodojo_dual_x5_capabilities_v72')
        with pytest.raises(ValueError,match='implicit CPU baseline'):
            h._validate_binding()
    finally:h.close('test')


def test_tcp_overlay_is_opt_in_keeps_original_and_archives_delivered_images(tmp_path):
    import base64
    import hashlib
    from io import BytesIO
    from PIL import Image
    h, remote = host(tmp_path, dual=True, visual_cloth=True)
    frame = remote.packet['observation']['cameras'][0]
    data = BytesIO();Image.new('RGB', (640,480)).save(data, format='PNG')
    frame['rgb_base64'] = base64.b64encode(data.getvalue()).decode()
    pose = np.eye(4);pose[2,3] = 1
    remote.packet['control']['tcp_poses'] = {'left': pose.tolist(), 'right': pose.tolist()}
    result = h.call('observe', {})
    assert len([b for b in result if b.type == 'image']) == 1
    assert 'visual_aids' not in json.loads(result[0].text)
    h.config = ConfigRegistry().resolve('robodojo_dual_x5_visual_cloth_v75')
    result = h.call('observe', {})
    images = [b for b in result if b.type == 'image']
    assert len(images) == 2 and images[0].data == frame['rgb_base64']
    value = json.loads(result[0].text)
    assert value['visual_aids'][0]['status'] == 'available'
    assert value['action_index'] == 0
    directory = h.output/'calls'/'000002'
    blocks = json.loads((directory/'response.json').read_text())['content']
    for image, block in zip(images, [b for b in blocks if b['type'] == 'image']):
        assert (directory/block['path']).read_bytes() == base64.b64decode(image.data)
        assert block['sha256'] == hashlib.sha256(base64.b64decode(image.data)).hexdigest()
    h.close('test')


@pytest.mark.parametrize('limit,version',[(2000,76),(4000,77)])
def test_tower_extended_budget_verifies_both_live_limits(tmp_path,limit,version):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        h.config=ConfigRegistry().resolve(f'robodojo_dual_x5_physical_visual_{limit}_v{version}')
        protocol={'override_id':f'build_tower_{limit}_v1','original_native_action_limit':1050,
                  'effective_native_action_limit':limit,'original_adapter_action_limit':1000,
                  'effective_adapter_action_limit':limit,'comparable_to_native_budget_baseline':False}
        limits=[{'source':source,'action_limit':limit} for source in ('robodojo_native_step_lim','openeta_adapter_max_episode_steps')]
        h.packet['control']['episode_budget']={'protocol':protocol,'limits':limits}
        h._validate_binding()
        for layer in limits:
            layer['action_limit']=1000
            with pytest.raises(ValueError,match='quotas'):h._validate_binding()
            layer['action_limit']=limit
    finally:h.close('test')


@pytest.mark.parametrize('preset,profile', [
    ('robodojo_dual_x5_completion_oracle_v78','build_tower_semantic_v1'),
    ('robodojo_dual_x5_coarse_completion_v80','robodojo_native_coarse_v1'),
])
def test_completion_oracle_is_opt_in_and_measured(tmp_path,preset,profile):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        assert 'check_task_completion' not in h.schemas
    finally:h.close('test')
    # Separate constructor using an actual opt-in config and a fake remote binding.
    config=ConfigRegistry().resolve(preset)
    r=Remote();r.packet=copy.deepcopy(h.packet)
    r.packet['observation']['task']='Build a tower'
    r.packet['control']['completion_feedback']=profile
    r.packet['control']['episode_budget']={'limits':[{'source':s,'action_limit':4000} for s in ('robodojo_native_step_lim','openeta_adapter_max_episode_steps')],
        'protocol':{'override_id':'build_tower_4000_v1','original_native_action_limit':1050,'effective_native_action_limit':4000,'original_adapter_action_limit':1000,'effective_adapter_action_limit':4000,'comparable_to_native_budget_baseline':False}}
    args=SimpleNamespace(output=tmp_path/'oracle',sim_url='http://example.invalid',env_id='openeta/robodojo_dual_x5_build_tower-v0',task='Build a tower',seed=0,timeout=300,max_requests=20)
    h=RoboDojoHost(args,config,config.bind_task(env_id=args.env_id,task=args.task,seed=0),transport=r)
    try:
        assert 'check_task_completion' in h.schemas
        r.packet['receipt']={'status':'checked','source_action_index':0,'native_actions':0,'physics_steps':0,'issues':[{'code':'robot_return','message':'Return not complete'}]}
        value=json.loads(h.call('check_task_completion',{})[0].text)
        assert value['receipt']['status']=='checked'
        assert r.calls[-1][1]['request']['operation']=='check_completion'
        assert h._status()['completion_checks']==1 and h.packet['action_index']==0
        assert r.calls[-1][1]['request']['completion_feedback']==profile
    finally:h.close('test')


def test_execution_horizon_forwarded_only_at_whole_request_scope(tmp_path):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        motion={'role':'left','target':{'delta_xyz':[0,0,.001]}}
        for name,args in [('move_to',{**motion,'max_duration_s':.1}),
                          ('move_to',{'motions':[motion],'max_duration_s':.2}),
                          ('gripper_control',{'role':'left','openness':0,'max_duration_s':.08})]:
            payload=json.loads(h.call(name,args)[0].text)
            assert payload.get('error') is None
            assert r.calls[-1][1]['request']['max_duration_s']==args['max_duration_s']
        before=len(r.calls)
        for args in [{**motion,'max_duration_s':0}, {**motion,'max_duration_s':2.1},
                     {'motions':[{**motion,'max_duration_s':.1}]}]:
            payload=json.loads(h.call('move_to',args)[0].text)
            assert payload['error']['code']=='request_rejected'
        assert len(r.calls)==before
    finally:h.close('test')


def test_execution_horizon_not_available_on_legacy_profile(tmp_path):
    h,r=host(tmp_path)
    try:
        before=len(r.calls)
        payload=json.loads(h.call('gripper_control',{'role':'primary','openness':0,'max_duration_s':.1})[0].text)
        assert payload['error']['code']=='request_rejected' and len(r.calls)==before
    finally:h.close('test')


def test_motion_feedback_separates_sampled_self_collision_from_other_arm_and_scene():
    from tools.codex_robodojo_host import motion_feedback
    part = lambda role, name: {'kind':'robot','role':role,'part':name,'private_asset':'not_public'}
    contacts = [{'participants':[part('right','link5'),part('right','link3')]},
                {'participants':[part('right','link5'),part('left','link3')]},
                {'participants':[part('right','link5'),{'kind':'scene_geometry','private_pose':[1,2,3]}]}]
    receipt={'native_actions':0,'physics_steps':0,'stop':{'reason':'requested_cartesian_path_not_certified',
             'cartesian_check':{'reason':'ik_candidates_in_collision','contacts':contacts}}}
    feedback=motion_feedback(receipt)
    assert feedback['reported_collision_kinds']==['inter_arm_collision','robot_scene_collision','self_collision']
    assert feedback['evidence_scope']=='tested_planning_configurations_no_physics'
    assert not feedback['scene_state_read'] and not feedback['target_changed']
    assert 'not_public' not in json.dumps(feedback) and 'private_pose' not in json.dumps(feedback)
    assert 'same arm' in feedback['guidance'][0]


def test_certificate_limiting_pairs_are_not_reported_as_collision_contacts():
    from tools.codex_robodojo_host import motion_feedback
    pair={'participants':[{'kind':'robot','role':'right','part':'link2'},
                          {'kind':'robot','role':'right','part':'base_link'}]}
    feedback=motion_feedback({'native_actions':0,'physics_steps':0,
        'stop':{'reason':'clearance_certificate_budget','contacts':[], 'clearance_limiting_pairs':[pair]}})
    assert feedback['certificate_budget_exhausted'] and feedback['clearance_limiting_pairs']
    assert not feedback['reported_collision_pairs'] and not feedback['reported_collision_kinds']


def test_successful_repair_does_not_relabel_old_failed_candidates_as_current_collision():
    from tools.codex_robodojo_host import motion_feedback
    receipt={'reached_target':True,'coordination':{'status':'safe','attempts':{
        'contacts':[{'participants':[{'kind':'robot','role':'left'},{'kind':'robot','role':'right'}]}]}}}
    assert motion_feedback(receipt) is None
    receipt['reached_target']=False;receipt['stop']={'reason':'requested_execution_horizon_reached'}
    assert motion_feedback(receipt) is None


def test_positive_gap_regression_is_not_summarized_as_collision():
    from tools.codex_robodojo_host import motion_feedback
    pair={'participants':[{'kind':'robot','role':'left','part':'camera'},{'kind':'scene_geometry','id':'geometry_224'}],
          'measured_signed_clearance_m':.00004}
    f=motion_feedback({'native_actions':1,'physics_steps':1,
        'stop':{'reason':'measured_positive_gap_not_preserved','detail':{'contacts':[pair]}}})
    assert f['positive_gap_regression_pairs'] and not f['reported_collision_pairs']
    assert not f['reported_collision_kinds'] and 'geometry_224' not in json.dumps(f)
    pair['measured_signed_clearance_m']=-.00004
    f=motion_feedback({'native_actions':1,'physics_steps':1,'stop':{'detail':{'contacts':[pair]}}})
    assert not f['positive_gap_regression_pairs'] and f['reported_collision_kinds']==['robot_scene_collision']


def test_contact_bound_summary_names_reported_finger_without_claiming_force_or_collision():
    from tools.codex_robodojo_host import motion_feedback
    pair={'participants':[{'kind':'robot','role':'left','part':'finger_1'},{'kind':'scene_geometry'}]}
    f=motion_feedback({'native_actions':1,'physics_steps':1,
        'stop':{'reason':'physical_contact_penetration_limit','detail':{'contact_bound_violations':[pair]}}})
    assert f['contact_bound_violation_pairs']==[pair['participants']]
    assert not f['reported_collision_kinds'] and not f['reported_collision_pairs']
    assert any('different arm' in g for g in f['guidance'])


def test_representative_collision_scope_survives_summary_without_extra_fields():
    from tools.codex_robodojo_host import motion_feedback
    pair={'participants':[{'kind':'robot','role':'right','part':'link5'},{'kind':'robot','role':'right','part':'link3'}]}
    context={'selection':'least_joint_displacement_among_tested_rejected_candidates','candidate_rank':1,
             'scope':'one_rejected_ik_configuration_not_all_candidates','private_unused_field':'omit'}
    f=motion_feedback({'native_actions':0,'physics_steps':0,
        'stop':{'contacts':[pair],'collision_evidence':context}})
    assert f['collision_evidence_contexts'][0]['candidate_rank']==1
    assert 'private_unused_field' not in json.dumps(f)


def test_unlimited_binding_requires_explicit_null_both_layers_and_original_quotas(tmp_path):
    from sim.envs.robodojo.episode_protocol import UNLIMITED_ACTIONS, bind_protocol
    from sim.envs.robodojo.worker_bridge import episode_budget
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        # Reuse the physical interface contract; only budget binding is under test.
        config=copy.deepcopy(h.config.value)
        config['components']['benchmark']['episode_budget_override']=UNLIMITED_ACTIONS
        h.config=SimpleNamespace(value=config,id=h.config.id)
        a=SimpleNamespace(_env=SimpleNamespace(step_lim=1600,take_action_cnt=[0]),
                          _max_episode_steps=1000,_step_count=0,_terminal=False,
                          _task_name='imitate_sorting_sequence',_robot_variant='dual_x5')
        bind_protocol(a,UNLIMITED_ACTIONS,'status')
        budget=episode_budget(a);h.packet['control']['episode_budget']=budget
        h._validate_binding()
        for mutation in [lambda b:b.pop('unlimited'),lambda b:b['limits'].pop(),
                         lambda b:b['limits'][0].pop('action_limit'),
                         lambda b:b['limits'][1].update(action_limit=1600),
                         lambda b:b['protocol'].pop('effective_adapter_action_limit')]:
            bad=copy.deepcopy(budget);mutation(bad);h.packet['control']['episode_budget']=bad
            with pytest.raises(ValueError,match='unlimited-action'):h._validate_binding()
    finally:h.close('test')


def test_no_episode_deadline_still_enforces_200_tool_requests(tmp_path):
    h,r=host(tmp_path)
    try:
        h.args.timeout=0;h.args.max_requests=200;h.started-=10**6;h.tool_calls=199
        result=json.loads(h.call('observe',{})[0].text)
        assert not result.get('error') and h.tool_calls==200 and not h.closed
        assert h._status()['limits']=={'max_requests':200,'episode_timeout_s':None}
        before=len(r.calls)
        result=json.loads(h.call('observe',{})[0].text)
        assert result['error'] and h.closed
        assert not any(n=='configured_robodojo' for n,_ in r.calls[before:])
    finally:h.close('test')


def test_initialization_binding_is_explicit_complete_and_forwarded(tmp_path):
    h,r=host(tmp_path,dual=True,physical=True)
    try:
        # Reuse the live packet's baseline budget; exercise only the init addition.
        value=h.config.value
        value['components']['benchmark']['initialization_profile']='egg_precheck_settle_300_v1'
        h.config=type(h.config)(json.dumps(value))
        with pytest.raises(ValueError,match='initialization'):h._validate_binding()
        receipt={'profile':'egg_precheck_settle_300_v1','completed_steps':300,
                 'native_stability_passed':True,'phase':'passed',
                 'comparable_to_native_initialization_baseline':False}
        h.packet['control']['initialization']=receipt
        h._validate_binding()
        for key,bad in [('completed_steps',299),('native_stability_passed',False),('phase','rejected')]:
            old=receipt[key];receipt[key]=bad
            with pytest.raises(ValueError,match='Initialization'):h._validate_binding()
            receipt[key]=old
        r.packet['control']['initialization']=copy.deepcopy(receipt)
        h._remote({'operation':'status'})
        assert r.calls[-1][1]['request']['initialization_profile']==receipt['profile']
        assert h._status()['initialization']==receipt
    finally:h.close()
