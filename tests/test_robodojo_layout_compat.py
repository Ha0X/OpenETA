import copy
import json
from types import SimpleNamespace

import pytest

from sim.envs.robodojo.layout import (COMPETITION, bind_layout, describe_layout,
                                     resolve_native_layout, task_robot_name, verify_live_layout)
from sim.envs.robodojo.upstream_compat import (install_action_codec, install_action_dimensions,
                                               install_robot_naming)


def document(*names):
    return {'robots':[{'robot_name':name, 'type':'support' if name == 'franka' else 'target'}
                      for name in names]}


def test_task_override_and_native_file_are_both_resolved():
    settings = {'common':{'robot_config':'dual_x5'},
                'tasks':{'opponent':{'robot_config':COMPETITION}}}
    assert task_robot_name('ordinary', settings) == 'dual_x5'
    seen = []
    def load(path):
        seen.append(path)
        return document('x5','x5','franka')
    native = resolve_native_layout('/native', 'opponent', settings, load)
    assert seen == ['/native/env_cfg/robot/'+COMPETITION+'.yml']
    assert native['target_roles'] == ['left','right']
    assert native['support_actors']['support_arm0']['agent_controllable'] is False
    with pytest.raises(ValueError, match='time_dependent_collision_check'):
        bind_layout(native, document('x5','x5'), 'dual_x5')


@pytest.mark.parametrize('robots', [[], ['x5'], [None],
    [{'robot_name':'x5','type':'support'}, {'robot_name':'x5'}],
    [{'robot_name':'x5'}, {'robot_name':'x5'}, {'robot_name':'franka','type':'target'}],
    [{'robot_name':'franka','type':'support'}, {'robot_name':'x5'}, {'robot_name':'x5'}]])
def test_malformed_or_misclassified_robots_fail_closed(robots):
    with pytest.raises(ValueError):
        describe_layout({'robots':robots})


def test_explicit_single_experiment_is_preserved_but_wrong_loaded_layout_is_rejected():
    native = describe_layout(document('x5','x5'))
    single = bind_layout(native, document('x5'), 'single_x5')
    assert single['experimental_override'] is True
    dual = bind_layout(native, document('x5','x5'), 'dual_x5')
    assert dual['experimental_override'] is False
    with pytest.raises(ValueError, match='differs'):
        bind_layout(native, document('x5'), 'dual_x5')
    with pytest.raises(ValueError):
        task_robot_name('task', {'common':{'robot_config':'../../secret'}})
    with pytest.raises(RuntimeError, match='count'):
        verify_live_layout(SimpleNamespace(), dual)
    env = SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[SimpleNamespace(type='target')]*2))
    verify_live_layout(env, dual)
    assert env._openeta_layout_contract == dual
    env.robot_manager.robot_list[0] = SimpleNamespace(type='support')
    with pytest.raises(RuntimeError, match='support'):
        verify_live_layout(env, dual)


def test_compatibility_hooks_are_idempotent_and_preserve_other_actions():
    class RobotManager:
        def restore_name(self, name):
            return 'original:'+name
    class Eval:
        def validate_action_dict(self, action):
            return action
        def get_action_type(self, action):
            return 'joint'
    for _ in range(2):
        install_robot_naming(RobotManager)
        install_action_codec(Eval)
    assert RobotManager().restore_name('ee_joint_state') == 'ee'
    assert RobotManager().restore_name('left_arm_joint_state') == 'original:left_arm_joint_state'
    action = {'arm_ee_pose':[1,2,3]}
    before = copy.deepcopy(action)
    assert Eval().validate_action_dict(action) == {'ee_pose':[1,2,3]}
    assert action == before
    assert Eval().get_action_type({'left_ee_pose':[]}) == 'ee'
    assert Eval().get_action_type({'left_arm_joint_state':[]}) == 'joint'
    module = SimpleNamespace(get_robot_action_dim_info=lambda _: None)
    install_action_dimensions(module, 'single_x5')
    assert module.get_robot_action_dim_info(None) == {'arm_dim':[6], 'ee_dim':[1]}
    install_action_dimensions(module, 'dual_x5')
    assert module.get_robot_action_dim_info(None) == {'arm_dim':[6,6], 'ee_dim':[1,1]}
    with pytest.raises(RuntimeError, match='missing'):
        install_action_codec(type('ChangedNativeAPI', (), {}))
    with pytest.raises(ValueError):
        install_action_dimensions(module, COMPETITION)


def test_admission_levels_never_imply_full_task_success():
    from scripts.robodojo_batch_acceptance import verification_levels
    levels = verification_levels()
    for name in ('layout','protocol','motion'):
        levels[name]['status'] = 'passed'
    assert levels['task']['status'] == 'not_run'
    assert all(v['status'] == 'not_run' for v in verification_levels().values())


def test_auxiliary_admission_is_explicit_and_preserves_native_layout_and_ownership():
    from sim.envs.robodojo.layout import SUPPORT_POLICY
    doc=document('x5','x5','franka');native=describe_layout(doc)
    contract=bind_layout(native,doc,'dual_x5',support_policy=SUPPORT_POLICY)
    assert contract['support_policy']==SUPPORT_POLICY
    assert contract['effective']['target_roles']==['left','right']
    env=SimpleNamespace(robot_manager=SimpleNamespace(robot_list=[SimpleNamespace(type=t)
        for t in ['target','target','support']]))
    verify_live_layout(env,contract)
    with pytest.raises(ValueError):bind_layout(native,doc,'dual_x5')
    with pytest.raises(ValueError):bind_layout(native,document('x5','x5'),'dual_x5',support_policy=SUPPORT_POLICY)
    changed=copy.deepcopy(doc);changed['robots'][2]['position']=[1,0,0]
    with pytest.raises(ValueError):bind_layout(native,changed,'dual_x5',support_policy=SUPPORT_POLICY)
    with pytest.raises(ValueError):bind_layout(native,doc,'single_x5',support_policy=SUPPORT_POLICY)
    env.robot_manager.robot_list[-1].type='target'
    with pytest.raises(RuntimeError,match='ownership'):verify_live_layout(env,contract)
