from types import SimpleNamespace

import pytest

from sim.envs.robodojo.calibration import SENSOR_CONTRACT
from sim.envs.robodojo.worker_bridge import operate


def setup_bridge():
    calls = []
    class DispatchReached(Exception):
        pass
    def move(targets, **kwargs):
        calls.append((targets, kwargs))
        raise DispatchReached
    control = SimpleNamespace(contact_grants={}, allowed_pairs=[], support_holds={},
        world=SimpleNamespace(), native=SimpleNamespace(robot_manager=SimpleNamespace(
            scene=SimpleNamespace(env_origins=[[1., 2., 3.]]))), move=move)
    adapter = SimpleNamespace(_env=SimpleNamespace(_openeta_sensor_contract=SENSOR_CONTRACT),
        _task_name="general_pickup", _state_valid=True, _terminal=False, _step_count=7, _max_episode_steps=1000,
        _checked_controller=control,
        _checked_marks={'table': {'xyz':[.1,.2,.3], 'source_action_index':7},
                        'block': {'xyz':[.2,.3,.4], 'source_action_index':7}})
    body = {'operation':'move', 'action_index':7, 'targets':{'left':[]},
            'tool_contact':{'mode':'stroke', 'surface_mark':'table',
                            'target_marks':[], 'target_handles':['workpiece_prior']}}
    return adapter, body, calls, DispatchReached


def test_bridge_resolves_only_fresh_marks_and_preserves_opaque_identity():
    a, body, calls, dispatched = setup_bridge()
    body['tool_contact']['target_marks'] = ['block']
    with pytest.raises(dispatched):
        operate(a, body)
    args = calls[0][1]
    assert args['tool_handles'] == ['workpiece_prior']
    assert args['support_point'].tolist() == [1.1,2.2,3.3]
    assert args['tool_points'][0].tolist() == [1.2,2.3,3.4]


@pytest.mark.parametrize('invalid', [
    {'target_marks':[{}]}, {'target_marks':['']}, {'target_handles':[{}]},
    {'target_handles':['']}, {'target_marks':['block']*8},
    {'surface_mark':'missing'}, {'target_marks':['missing']},
])
def test_bridge_rejects_bad_contact_inputs_before_dispatch(invalid):
    a, body, calls, _ = setup_bridge()
    body['tool_contact'].update(invalid)
    with pytest.raises(ValueError):
        operate(a, body)
    assert not calls


def test_identity_reference_still_requires_current_table_observation():
    a, body, calls, _ = setup_bridge()
    a._checked_marks['table']['source_action_index'] = 6
    with pytest.raises(ValueError, match='fresh marked table'):
        operate(a, body)
    assert not calls


@pytest.mark.parametrize('mode', ['adjust', 'reorient'])
def test_reorientation_keeps_fresh_surface_and_explicit_workpiece_contract(mode):
    a, body, calls, dispatched = setup_bridge()
    body['tool_contact']['mode'] = mode
    with pytest.raises(dispatched):
        operate(a, body)
    assert calls[0][1]['tool_mode'] == mode
    assert calls[0][1]['tool_handles'] == ['workpiece_prior']
    a._checked_marks['table']['source_action_index'] = 6
    with pytest.raises(ValueError, match='fresh marked table'):
        operate(a, body)
    assert len(calls) == 1


def test_push_resolves_fresh_marks_without_creating_grasp_authorization():
    import numpy as np
    a,body,calls,dispatched=setup_bridge()
    body.pop('tool_contact');body['targets']={'left':[np.eye(4).tolist()]}
    body['push_contact']={'body_mark':'block','surface_mark':'table','target_marks':[],
                          'fingers':['finger_2'],'patch_radius_m':.3}
    with pytest.raises(dispatched):operate(a,body)
    spec=calls[0][1]['push_contact']
    assert spec['body_point'].tolist()==[1.2,2.3,3.4]
    assert spec['surface_point'].tolist()==[1.1,2.2,3.3]
    assert spec['fingers']==['finger_2'] and not a._checked_controller.contact_grants
    a._checked_marks['block']['source_action_index']=6
    with pytest.raises(ValueError,match='fresh visible'):operate(a,body)
    assert len(calls)==1


def test_invalid_horizon_does_not_mutate_legacy_contact_authorization():
    a,body,calls,_=setup_bridge()
    c=a._checked_controller
    c.authorize_contact=lambda *args,**kw:(_ for _ in ()).throw(AssertionError('must validate ceiling before authorizing'))
    body.update(max_duration_s=.1,contact_mark='block',role='left')
    previous=dict(c.contact_grants)
    with pytest.raises(ValueError,match='physical-contact'):operate(a,body)
    assert c.contact_grants==previous and not calls


def test_execution_horizon_context_is_cleared_after_dispatch_error():
    a,body,_,dispatched=setup_bridge()
    with pytest.raises(dispatched):operate(a,body)
    assert a._checked_controller._execution_horizon is None
