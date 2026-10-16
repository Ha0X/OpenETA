from types import SimpleNamespace
import copy
import json
import pytest
from sim.envs.robodojo import completion_feedback_generic as f


def leaf(name='is_axis_up'):
    return (name, {'label':'PRIVATE_OBJECT', 'threshold':.023, 'xyz':[1,2,3]})


def manager():
    return SimpleNamespace(check_list=[[]], final_check_list=[[]], trigger_check_list=[[]],
        query_list=[[]], trigger_query_list=[[]], score_list=[[leaf()]], num_envs=1)


def summary(m, trace=None, index=1, valid=True, success=False):
    return f.summarize(m,trace,index,official_success=success,episode_valid=valid)


def test_pending_is_not_geometry_failure_and_scores_are_not_success_conditions():
    m=manager();m.check_list=[[[leaf()]]]
    before=copy.deepcopy(vars(m));a=summary(m);assert a==summary(m)
    assert vars(m)==before
    assert a['issues'][0]['evidence']=='pending_not_evaluated'
    assert not a['native_requirements_recorded']
    m.check_list=[[]]
    assert summary(m)['native_requirements_recorded']
    assert summary(m)['issues']==[] # score entries do not block reward


def test_alternatives_remain_one_group_and_future_stages_stay_private():
    node=[[leaf(),leaf('is_A_depth_in_B')],[leaf('is_A_in_B')]]
    m=manager();m.check_list=[[[node],[leaf('is_garment_pointA_close_to_pointB_by_x_range')]]]
    t=f.NativeTrace();t.rows[id(node)]=(node,False);t.seal(1)
    result=summary(m,t);assert len(result['issues'])==1
    issue=result['issues'][0];assert issue['relations']==['insertion','orientation','placement']
    assert issue['evidence']=='last_native_evaluation'
    assert 'cloth' not in json.dumps(result)
    assert summary(m,t,index=2)['issues'][0]['evidence']=='pending_not_evaluated'
    t.rows[id(node)]=(node,True)
    assert summary(m,t)['issues']==[]
    assert not summary(m,t)['native_requirements_recorded'] # queue not consumed by us


def test_history_and_trigger_edges_are_not_recomputed():
    m=manager();m.trigger_check_list=[[([leaf('is_all_gripper_open')],[leaf('is_A_fluid_in_B')],'rising_edge',True)]]
    m.query_list=[[[[leaf('is_lift')],0,1]]]
    m.trigger_query_list=[[([], [leaf('is_joint_position_ratio_change_from_above_to_below')], 'rising_edge',False,3,1)]]
    result=summary(m,valid=False);codes={x['code'] for x in result['issues']}
    assert {'history_violation','history_requirement_pending','trigger_pending','episode_invalidated'}<=codes
    assert not result['native_requirements_recorded']
    encoded=json.dumps(result)
    for private in ('PRIVATE_OBJECT','0.023','xyz','threshold','rising_edge'):
        assert private not in encoded


def test_passive_trace_does_not_add_calls_and_restores_overrides_on_error():
    class Manager:
        def __init__(self):self.calls=0;self.counter=0
        def check_once(self,node,env_idx,op='or'):
            self.calls+=1
            if isinstance(node,list):return any(self.check_once(n,env_idx) for n in node)
            self.counter+=1
            return node[0]=='is_axis_up'
        def step(self):
            return self.check_once(node,0)
    node=[leaf('is_A_in_B'),leaf()]
    baseline=Manager();assert baseline.step()
    m=Manager();t=f.NativeTrace()
    with t.observe(m):assert m.step()
    t.seal(1)
    assert (m.calls,m.counter)==(baseline.calls,baseline.counter)
    assert t.verdict(node,1) is True and len(t.rows)==1
    assert 'step' not in vars(m) and 'check_once' not in vars(m)
    orig=m.check_once;m.check_once=orig
    with pytest.raises(RuntimeError):
        with t.observe(m):raise RuntimeError('physical failure')
    assert m.check_once is orig and not t.valid


def test_new_native_cycle_clears_previous_results():
    node=leaf()
    class Manager:
        def check_once(self,node,env_idx):return False
        def step(self,run=True):
            if run:self.check_once(node,0)
    m=Manager();t=f.NativeTrace()
    with t.observe(m):m.step();m.step(False)
    t.seal(1);assert t.verdict(node,1) is None


def test_unknown_revision_errors_are_sanitized_and_never_award_success(monkeypatch):
    m=manager();e=SimpleNamespace(end_flag=[False],success=[True],reward_manager=m)
    a=SimpleNamespace(_step_count=1,_env=e,_state_valid=True)
    def broken(_):raise ValueError('PRIVATE_OBJECT xyz threshold')
    monkeypatch.setattr(f,'_verify',broken)
    result=f.check(a,1);assert result['status']=='unavailable'
    assert 'PRIVATE_OBJECT' not in json.dumps(result)
    with pytest.raises(ValueError):f.check(a,0)
    with pytest.raises(ValueError):f.check(a,True)
    monkeypatch.setattr(f,'_verify',lambda _:None)
    result=f.check(a,1);assert result['native_requirements_recorded']
    assert result['current_conditions_satisfied'] is None
    assert not result['official_success_recorded'] and e.end_flag==[False]
    assert result['native_actions']==result['physics_steps']==0
    m.check_list=[[[('unknown_private_predicate',{})]]]
    assert f.check(a,1)['status']=='unavailable'


def test_registry_contains_exactly_42_pinned_base_tasks():
    from pathlib import Path
    inv=json.loads(Path('configs/harness/robodojo_rigid32_inventory.json').read_text())
    assert f.TASK_PINS=={t['task']:t['task_source_sha256'] for t in inv['tasks']}
    assert len(f.TASK_PINS)==42


def test_truncation_retains_irrecoverable_failure_first():
    m=manager();m.check_list=[[[leaf(names[0]) for names in f.FAMILIES.values()]]]
    m.query_list=[[[[leaf('is_lift')],0,1]]]
    result=summary(m,valid=False)
    assert result['issues_truncated'] and len(result['issues'])==8
    assert result['issues'][0]['code']=='episode_invalidated'
    assert result['issues'][1]['code']=='history_violation'


def turn_taking_nodes():
    def away(arm):
        return ('is_robot_not_back_to_origin',
                {'arm_tag': arm, 'pos_threshold': .31234567, 'rot_threshold': 31.234567, 'env_idx': 0})
    return [[away('left_arm'), away('right_arm')], away('support_arm0')]


def test_turn_taking_failure_is_explicit_read_only_and_keeps_arguments_private():
    m=manager();m.query_list=[[[turn_taking_nodes(),0,1]]]
    before=copy.deepcopy(vars(m));result=summary(m,valid=False)
    issue=next(i for i in result['issues'] if i['code']=='history_violation')
    assert 'turn-taking rule was violated' in issue['message']
    assert 'at the same time' in issue['message']
    assert 'not a collision' in issue['message']
    assert vars(m)==before
    encoded=json.dumps(result)
    for private in ('left_arm','right_arm','support_arm0','pos_threshold','rot_threshold','31234567','31.234567'):
        assert private not in encoded
    assert summary(m,success=True)['issues']==[]


@pytest.mark.parametrize('variant', ['unviolated','nonzero_target','flattened','positive_predicate','different_actor','extra_argument','triggered'])
def test_other_robot_return_queries_are_not_mislabeled_as_turn_taking(variant):
    nodes=turn_taking_nodes();target=0;count=1;m=manager()
    if variant=='unviolated':count=0
    elif variant=='nonzero_target':target=1;count=2
    elif variant=='flattened':nodes=nodes[0]+[nodes[1]]
    elif variant=='positive_predicate':nodes[1]=('is_robot_back_to_origin',nodes[1][1])
    elif variant=='different_actor':nodes[1][1]['arm_tag']='unreviewed_actor'
    elif variant=='extra_argument':nodes[1][1]['extra']='PRIVATE'
    if variant=='triggered':m.trigger_query_list=[[([],nodes,'level',True,target,count)]]
    else:m.query_list=[[[nodes,target,count]]]
    assert 'turn-taking' not in json.dumps(summary(m))


def test_kong_feedback_distinguishes_evaluated_requirements_without_private_targets():
    used=('is_axis_up', {'label':['mahjong0_1'], 'axis':[0,0,1], 'threshold':31.234567})
    rest=('is_axis_up', {'label':['mahjong1_2'], 'axis':[0,1,0], 'threshold':7.234567})
    intermediate=('is_qpos_close', {'label_A':'mahjong9_0', 'label_B':None, 'qpos':[0,.707,.707,0], 'dis_threshold':7.234567})
    placement=('is_A_xy_distance_close_to_pos', {'label':'mahjong9_0','pos':[.319,-.15],'dis_threshold':.015})
    m=manager();m.check_list=[[[used,rest,intermediate],[placement]]]
    t=f.NativeTrace();t.rows[id(used)]=(used,True);t.rows[id(rest)]=(rest,False);t.seal(5)
    before=copy.deepcopy(vars(m))
    r=f.summarize(m,t,5,official_success=False,episode_valid=True,task_name='make_kong')
    assert len(r['issues'])==2
    assert 'remaining checked hand tiles' in r['issues'][0]['message']
    assert r['issues'][0]['code']=='stage_not_met'
    assert 'intermediate orientation' in r['issues'][1]['message']
    assert r['issues'][1]['code']=='stage_pending'
    encoded=json.dumps(r)
    for private in ('mahjong','qpos','31.234567','7.234567','0.707','0.319','0.015','final placement'):
        assert private not in encoded
    assert vars(m)==before
    stale=f.summarize(m,t,6,official_success=False,episode_valid=True,task_name='make_kong')
    assert all(i['code']=='stage_pending' for i in stale['issues'])
    assert 'replacement tile' not in json.dumps(summary(m,t,index=5))


def test_kong_second_stage_placement_feedback_does_not_reveal_values_or_future_stages():
    node=('is_A_xy_distance_close_to_pos', {'label':'mahjong9_0','pos':[.319,-.15],'dis_threshold':.015})
    m=manager();m.check_list=[[[node]]];t=f.NativeTrace();t.rows[id(node)]=(node,False);t.seal(1)
    r=f.summarize(m,t,1,official_success=False,episode_valid=True,task_name='make_kong')
    assert 'replacement tile final placement' in r['issues'][0]['message']
    assert r['issues'][0]['relations']==['alignment']
    assert '0.319' not in json.dumps(r) and 'mahjong9' not in json.dumps(r)
    assert f._kong_requirement([node,leaf()]) is None
    assert f._kong_requirement(('is_axis_up',{'label':['other0'],'axis':[0,1,0]})) is None


TABLE_REQUIREMENTS = [
    ('is_not_moved', {'label': 'mouse'}, 'mouse resting stably'),
    ('is_A_in_B', {'label_A': 'mouse', 'label_B': 'mousemat'}, 'mouse placement'),
    ('is_axis_aligned', {'label_A': 'mouse', 'axis_A': [0,1,0], 'world_axis': [1,0,0]}, 'mouse required heading'),
    ('is_not_moved', {'label': 'keyboard'}, 'keyboard resting stably'),
    ('is_A_bbox_cover_rect_region', {'label_A': 'keyboard'}, 'keyboard coverage'),
    ('is_axis_aligned', {'label_A': 'keyboard', 'axis_A': [0,1,0], 'world_axis': [0,1,0]}, 'keyboard required heading'),
    ('is_axis_up', {'label': 'keyboard', 'axis': [0,0,1]}, 'keyboard face-up'),
    ('is_not_moved', {'label': 'garage'}, 'figurine resting stably'),
    ('is_A_up_B', {'label_A': 'garage', 'label_B': 'cube_cushion'}, 'figurine height'),
    ('is_axis_up', {'label': 'garage', 'axis': [0,0,1]}, 'figurine upright'),
    ('is_AB_xy_distance_within_threshold', {'label_A': 'garage', 'label_B': 'cube_cushion'}, 'figurine centering'),
    ('is_not_moved', {'label': 'alarm'}, 'alarm clock resting stably'),
    ('is_A_up_B', {'label_A': 'alarm', 'label_B': 'drawer'}, 'alarm clock height'),
    ('is_axis_up', {'label': 'alarm', 'axis': [0,0,1]}, 'alarm clock upright'),
    ('all_robot_back_to_origin', {}, 'both robot arms returning'),
]


@pytest.mark.parametrize('name,args,public_name', TABLE_REQUIREMENTS)
def test_table_feedback_identifies_each_native_requirement_without_values(name,args,public_name):
    args = dict(args, threshold=7.234567, private_target=[.319876, .123456])
    node = (name,args);m=manager();m.check_list=[[[node]]]
    t=f.NativeTrace();t.rows[id(node)]=(node,False);t.seal(9)
    before=copy.deepcopy(vars(m))
    result=f.summarize(m,t,9,official_success=False,episode_valid=True,task_name='organize_table')
    issue=result['issues'][0]
    assert public_name in issue['message'] and issue['code']=='stage_not_met'
    assert issue['evidence']=='last_native_evaluation'
    encoded=json.dumps(result)
    for private in ('7.234567','.319876','private_target','threshold','garage','cube_cushion','axis_A'):
        assert private not in encoded
    assert vars(m)==before
    pending=f.summarize(m,t,10,official_success=False,episode_valid=True,task_name='organize_table')
    assert pending['issues'][0]['code']=='stage_pending'
    assert public_name in pending['issues'][0]['message']
    assert 'do not infer a current geometric failure' in pending['issues'][0]['message']
    assert public_name not in json.dumps(summary(m,t,index=9)) # scoped to this task
    t.rows[id(node)]=(node,True)
    assert f.summarize(m,t,9,official_success=False,episode_valid=True,task_name='organize_table')['issues']==[]


def test_table_failed_clock_does_not_claim_unchecked_robot_return_or_recheck_geometry():
    clock=('is_axis_up',{'label':'alarm','axis':[0,0,1]})
    home=('all_robot_back_to_origin',{})
    m=manager();m.check_list=[[[clock,home]]]
    t=f.NativeTrace();t.rows[id(clock)]=(clock,False);t.seal(42)
    before=copy.deepcopy(vars(m))
    def run():return f.summarize(m,t,42,official_success=False,episode_valid=True,task_name='organize_table')
    a=run();assert a==run() and vars(m)==before
    assert [x['code'] for x in a['issues']]==['stage_not_met','stage_pending']
    assert 'alarm clock upright' in a['issues'][0]['message']
    assert 'both robot arms' in a['issues'][1]['message']
    assert not a['native_requirements_recorded']
    assert f.summarize(m,t,42,official_success=True,episode_valid=True,task_name='organize_table')['issues']==[]


@pytest.mark.parametrize('node', [
    [('is_axis_up',{'label':'alarm','axis':[0,0,1]})],
    ('is_axis_up',{'label':'PRIVATE_OBJECT','axis':[0,0,1]}),
    ('is_axis_up',{'label':['alarm'],'axis':[0,0,1]}),
    ('is_axis_up',{'label':'alarm','axis':[0,1,0]}),
    ('is_A_up_B',{'label_A':'alarm','label_B':'unreviewed_surface'}),
    ('is_axis_aligned',{'label_A':'mouse','axis_A':[0,1,0],'world_axis':[0,1,0]}),
])
def test_table_ambiguous_groups_and_unreviewed_selectors_remain_generic(node):
    assert f._organize_table_requirement(node) is None


@pytest.mark.parametrize('task,node,public', [
    ('fasten_screws',('is_A_depth_in_B',{'label_A':'nut1','label_B':'bolt1'}),'nut seated'),
    ('fold_clothes',('is_garment_pointA_close_to_pointB_by_y_range',{'label':'target','point_A':'left_sleeve','point_B':'right_chest'}),'sleeves tucked'),
    ('pour_liquid_into_cup',('is_axis_up',{'label':'bottle'}),'upright check trigger'),
    ('make_toast',('is_N_A_in_B',{'label_B':'bread_shelf'}),'two remaining slices'),
    ('sweep_blocks',('is_A_on_B_left',{'label_A':'broom_shovel','label_B':'broom'}),'dustpan left'),
    ('store_tools_in_toolbox',('is_A_cover_B',{'label_A':'toolbox','label_B':'hammer'}),'hammer fully contained'),
    ('swap_blocks',('is_joint_position_above_ratio',{'label':'button0'}),'fully released'),
])
def test_v111_task_feedback_preserves_read_only_evidence_and_sanitization(task,node,public):
    node=(node[0],dict(node[1],threshold=.1234567,private_pose=[.9876543,2,3]))
    m=manager();m.check_list=[[[node]]];t=f.NativeTrace();t.rows[id(node)]=(node,False);t.seal(1)
    before=copy.deepcopy(vars(m))
    def run(index):return f.summarize(m,t,index,official_success=False,episode_valid=True,task_name=task)
    a=run(1);assert public in a['issues'][0]['message'];assert a['issues'][0]['code']=='stage_not_met'
    assert run(2)['issues'][0]['code']=='stage_pending'
    assert a==run(1) and vars(m)==before
    for secret in ('private_pose','threshold','1234567','9876543'):
        assert secret not in json.dumps(a)
    from sim.envs.robodojo.completion_task_requirements import requirement
    assert requirement(task,[node]) is None
    assert requirement('not_reviewed',node) is None
