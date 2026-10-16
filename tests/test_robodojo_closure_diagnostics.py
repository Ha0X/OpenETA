import copy

import pytest

from sim.envs.robodojo.closure_diagnostics import closure_diagnostics


def measured(position=0., sign=1):
    return {
        'measurement_available': True,
        'measurement_source': 'native_joint_pos',
        'openness_basis': 'first_actuated_joint_physical_limits',
        'normalization': {'joint_name': 'finger', 'physical_joint_limits': [0., .044],
                          'command_joint_limits': [-.01, .044], 'sign': sign},
        'joint_positions': {'finger': position},
    }


@pytest.mark.parametrize('position,sign', [(0., 1), (.044, -1)])
def test_closed_physical_limit_cannot_be_grasp_confirmation(position, sign):
    r = closure_diagnostics(measured(position, sign), 0.)
    assert r['status'] == 'near_closed_limit'
    assert r['remaining_closing_travel_fraction'] == 0
    assert not r['grasp_confirmed']
    assert r['load_bearing_support'] == 'not_established_by_joint_position'
    assert 'external support' in r['guidance']
    assert r['scope'] == 'first_actuated_joint_only'


@pytest.mark.parametrize('position,sign', [(.01, 1), (.034, -1)])
def test_opening_does_not_claim_object_contact_or_force(position, sign):
    r = closure_diagnostics(measured(position, sign), 0.)
    assert r['status'] == 'above_closed_limit'
    assert r['remaining_closing_travel_fraction'] == pytest.approx(.01/.044)
    assert 'tracking lag' in r['guidance']
    assert 'not a force measurement' in r['guidance']
    assert not r['grasp_confirmed']


def test_tolerance_is_reported_and_not_a_binary_grasp_classifier():
    assert closure_diagnostics(measured(.00008), 0.)['status'] == 'near_closed_limit'
    r = closure_diagnostics(measured(.0001), 0.)
    assert r['status'] == 'above_closed_limit'
    assert r['closed_limit_tolerance_fraction'] == .002
    assert not r['grasp_confirmed']


@pytest.mark.parametrize('openness', [.1, .5, 1.])
def test_partial_or_open_request_does_not_assess_closure(openness):
    r = closure_diagnostics(measured(), openness)
    assert r['status'] == 'not_requested'
    assert 'remaining_closing_travel_fraction' not in r


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -.0001, .0441])
def test_invalid_joint_value_stays_unknown(value):
    assert closure_diagnostics(measured(value), 0.)['status'] == 'unavailable'


@pytest.mark.parametrize('change', [
    {'measurement_available': False},
    {'openness_basis': 'first_actuated_joint_normalized'},
    {'normalization': {}},
    {'normalization': {'joint_name': 'finger', 'physical_joint_limits': [1, 1], 'sign': 1}},
    {'normalization': {'joint_name': 'finger', 'physical_joint_limits': [0, .044], 'sign': True}},
    {'joint_positions': {}},
])
def test_unverified_or_incomplete_measurement_stays_unknown(change):
    v = measured(); v.update(change)
    assert closure_diagnostics(v, 0.)['status'] == 'unavailable'


def test_feedback_does_not_mutate_original_measurement():
    v = measured(); before = copy.deepcopy(v)
    closure_diagnostics(v, 0.)
    assert v == before
