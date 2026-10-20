from types import SimpleNamespace
import copy
import json
import numpy as np
import pytest
from sim.envs.robodojo import completion_feedback as f


class Parser:
    @staticmethod
    def is_axis_up(snapshot,args):
        return float(snapshot.pass_labels[args['label']])


def leaf(name):return ('is_axis_up',{'label':name})


def test_native_alternating_and_or_semantics():
    s=SimpleNamespace(pass_labels={'a':True,'b':False,'c':True})
    # OR of two AND alternatives, like the native base support choices.
    tree=[[leaf('a'),leaf('b')],[leaf('a'),leaf('c')]]
    assert f._node(tree,Parser,s)
    s.pass_labels['c']=False
    assert not f._node(tree,Parser,s)
    with pytest.raises(ValueError):f._node(('get_reward',{}),Parser,s)


def test_all_issues_semantic_and_no_predicate_args_returned():
    s=SimpleNamespace(pass_labels={'block3':False,'arm_private':False})
    groups=[('centering','The thin plank is off-center.',[leaf('block3')]),
            ('return','One or more arms have not returned.',[leaf('arm_private')])]
    satisfied,issues=f.evaluate_groups(groups,[leaf('block3'),leaf('arm_private')],Parser,s)
    assert not satisfied and len(issues)==2
    assert 'block3' not in json.dumps(issues) and 'arm_private' not in json.dumps(issues)
    s.pass_labels={'block3':True,'arm_private':True}
    assert f.evaluate_groups(groups,[leaf('block3'),leaf('arm_private')],Parser,s)==(True,[])


def test_fail_closed_and_stale_checks_do_not_read_live_state(monkeypatch):
    env=SimpleNamespace(end_flag=[False],success=[True])
    a=SimpleNamespace(_env=env,_step_count=7,_task_name='build_tower',_robot_variant='dual_x5',_reward_registered=True)
    def broken(_):raise ValueError('PRIVATE block3 xyz=[1,2,3] threshold=0.023')
    monkeypatch.setattr(f,'_verify',broken)
    with pytest.raises(ValueError,match='Stale'):f.check(a,6)
    result=f.check(a,7)
    assert result['status']=='unavailable' and result['current_conditions_satisfied'] is None
    assert 'PRIVATE' not in json.dumps(result) and 'block3' not in json.dumps(result)
    assert a._step_count==7 and env.end_flag==[False] and env.success==[True]


def test_satisfied_snapshot_does_not_award_official_success(monkeypatch):
    parser=Parser();env=SimpleNamespace(end_flag=[False],success=[True],reward_manager=SimpleNamespace(func_parser=parser))
    a=SimpleNamespace(_env=env,_step_count=7,_task_name='build_tower',_robot_variant='dual_x5',_reward_registered=True)
    monkeypatch.setattr(f,'_verify',lambda _:None)
    monkeypatch.setattr(f,'_groups',lambda _:([('orientation','Bad orientation',[leaf('a')])],[leaf('a')]))
    monkeypatch.setattr(f,'_snapshot',lambda _:SimpleNamespace(pass_labels={'a':True}))
    first=f.check(a,7);second=f.check(a,7)
    assert first==second and first['current_conditions_satisfied']
    assert not first['official_success_recorded'] and first['native_actions']==first['physics_steps']==0
    assert env.end_flag==[False]


def test_frozen_snapshot_does_not_read_later_live_changes():
    class Layout:
        def __init__(self):self.pos=np.zeros(3)
        def get_instance_name(self,**kw):return kw['label']
        def get_scene_object(self,**kw):return object()
        def get_instance_pose(self,**kw):return self.pos,[1,0,0,0]
        def get_instance_metadata(self,**kw):return {}
        def get_support_points(self,**kw):return [[0,0,.01,1,0,0,0]],[.02]
    live=Layout();frozen=f.FrozenLayout(live);live.pos[0]=9
    pos,_=frozen.get_instance_pose(inst_name='block0',env_idx=0)
    assert pos[0]==0
    pos[0]=8
    assert frozen.get_instance_pose(inst_name='block0',env_idx=0)[0][0]==0
