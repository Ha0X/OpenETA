import json
from types import SimpleNamespace
from sim.envs.robodojo.checked_control import CheckedController


def test_positive_margin_failure_names_limiting_parts_without_private_paths():
    c=CheckedController.__new__(CheckedController);c.roles=['left']
    c.world=SimpleNamespace(table_id='/World/private/table',scene=SimpleNamespace(snapshot={'robots':[{'role':'left'}]},items=[
        {'id':'/World/private/robot/finger','robot_index':0,'body':'link7'},
        {'id':'/World/private/table','robot_index':None,'body':None}]))
    r=c._public_check({'status':'blocked','reason':'insufficient_clearance','detail':{
        'contacts':[],'clearance_m':.00019,
        'closest_pair':('/World/private/robot/finger','/World/private/table'),
        'uncertified_pairs':[(0,1)]}})
    assert r['contacts']==[]
    assert r['closest_pair']['participants']==[
        {'kind':'robot','role':'left','part':'finger_1'},
        {'kind':'static_support','part':'table'}]
    assert r['clearance_limiting_pair_count']==1
    assert '/World' not in json.dumps(r)
    assert 'No overlap' in r['guidance']


def test_domain_guidance_is_not_replaced_by_generic_budget_advice():
    c=CheckedController.__new__(CheckedController)
    result=c._public_check({'status':'unknown', 'reason':'ik_candidates_in_collision',
        'guidance':'Finite candidate search is not proof that this pose is unreachable.'})
    assert result['guidance'].startswith('Finite candidate search')


def test_nested_recovery_proof_budget_is_reported_without_contacts():
    c=CheckedController.__new__(CheckedController)
    r=c._public_check({'status':'unknown','reason':'positive_gap_swept_nonintersection_not_certified',
        'detail':{'status':'unknown','reason':'path_check_budget','samples':1024}})
    assert r['detail']=={'status':'unknown','reason':'path_check_budget','samples':1024}
    assert r['status']=='unknown' and 'guidance' in r


def test_nested_recovery_collision_keeps_cause_and_anonymizes_geometry():
    c=CheckedController.__new__(CheckedController);c.roles=['left']
    c.world=SimpleNamespace(table_id='/World/private/table',scene=SimpleNamespace(snapshot={'robots':[{'role':'left'}]},items=[
        {'id':'/World/private/finger','robot_index':0,'body':'link7'},
        {'id':'/World/private/table','robot_index':None,'body':None}]))
    r=c._public_check({'status':'blocked','reason':'positive_gap_swept_nonintersection_not_certified',
        'detail':{'status':'blocked','reason':'geometry_collision','fraction':.25,
            'detail':{'contacts':[{'a':'/World/private/finger','b':'/World/private/table',
                'robot_a':0,'robot_b':None}],'clearance_m':0.}}})
    assert r['detail']['reason']=='geometry_collision' and r['detail']['fraction']==.25
    assert r['detail']['contacts'][0]['participants'][0]['part']=='finger_1'
    assert '/World' not in json.dumps(r)


def test_marked_object_feedback_links_existing_authorization_without_grasp_oracle():
    c=CheckedController.__new__(CheckedController);c.roles=['left','right']
    c.world=SimpleNamespace(table_id='table',scene=SimpleNamespace(items=[
        {'id':'/World/private/object','robot_index':None,'body':None},
        {'id':'/World/private/unmarked','robot_index':None,'body':None}]))
    c.contact_grants={'left':{'indices':[0],'root':'/World/private/object'}}
    assert c._part(0)=={'kind':'scene_geometry','id':'geometry_0',
        'contact_authorizations':[{'role':'left','contact_id':'contact_0'}], 'grasp_confirmed':False}
    assert c._part(1)=={'kind':'scene_geometry','id':'geometry_1'}


def test_episode_budget_reports_actual_adapter_limit_without_changing_it():
    from sim.envs.robodojo.worker_bridge import episode_budget
    adapter=SimpleNamespace(_max_episode_steps=1000,_step_count=790,_terminal=False)
    before=vars(adapter).copy()
    result=episode_budget(adapter)
    assert result['remaining_actions']==210 and result['action_limit']==1000
    assert vars(adapter)==before and result['terminal'] is False
    adapter._step_count=1001;adapter._terminal=True
    assert episode_budget(adapter)['remaining_actions']==0


def test_episode_budget_reports_native_exhaustion_without_modifying_quotas():
    from sim.envs.robodojo.worker_bridge import episode_budget
    native=SimpleNamespace(step_lim=200,take_action_cnt=[200])
    adapter=SimpleNamespace(_max_episode_steps=1000,_step_count=200,_terminal=True,_env=native)
    result=episode_budget(adapter)
    assert result['source']=='robodojo_native_step_lim'
    assert result['action_limit']==200 and result['used_actions']==200
    assert result['remaining_actions']==0 and result['native_limit_available']
    assert result['limits'][0]['remaining_actions']==800
    assert native.step_lim==200 and native.take_action_cnt==[200]
    assert adapter._max_episode_steps==1000


def test_episode_budget_uses_each_layers_own_counter_and_stricter_remaining():
    from sim.envs.robodojo.worker_bridge import episode_budget
    native=SimpleNamespace(step_lim=800,take_action_cnt=[794])
    adapter=SimpleNamespace(_max_episode_steps=1000,_step_count=790,_terminal=False,_env=native)
    result=episode_budget(adapter)
    assert result['remaining_actions']==6 and result['used_actions']==794
    adapter._max_episode_steps=795
    result=episode_budget(adapter)
    assert result['remaining_actions']==5 and result['used_actions']==790
    assert result['source']=='openeta_adapter_max_episode_steps'


def test_episode_budget_does_not_guess_missing_or_ambiguous_native_counters():
    from sim.envs.robodojo.worker_bridge import episode_budget
    for counts in (None,[],[1,2],[float('nan')],[-1],[True]):
        adapter=SimpleNamespace(_max_episode_steps=1000,_step_count=10,_terminal=False,
                                _env=SimpleNamespace(step_lim=800,take_action_cnt=counts))
        result=episode_budget(adapter)
        assert not result['native_limit_available']
        assert result['remaining_actions']==990
