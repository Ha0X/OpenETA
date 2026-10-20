"""Explain public gripper travel without inferring object contact or support."""
import math


def closure_diagnostics(measured, requested_openness):
    result = {
        'status': 'not_requested',
        'scope': 'first_actuated_joint_only',
        'grasp_confirmed': False,
        'load_bearing_support': 'not_established_by_joint_position',
    }
    if not 0 <= requested_openness <= .05:
        return result
    result['status'] = 'unavailable'
    if (measured.get('measurement_available') is not True or
            measured.get('openness_basis') != 'first_actuated_joint_physical_limits'):
        result['reason'] = 'verified_physical_joint_measurement_required'
        return result
    try:
        mapping = measured['normalization']
        name = mapping['joint_name']
        low, high = map(float, mapping['physical_joint_limits'])
        sign = mapping['sign']
        position = float(measured['joint_positions'][name])
        if (sign not in (-1, 1) or isinstance(sign, bool) or
                not all(map(math.isfinite, (low, high, position))) or
                high <= low or not low <= position <= high):
            raise ValueError('invalid measured joint or physical limits')
    except (KeyError, TypeError, ValueError):
        result['reason'] = 'invalid_physical_joint_mapping'
        return result
    distance = position-low if sign == 1 else high-position
    fraction = distance/(high-low)
    tolerance = .002
    at_limit = fraction <= tolerance
    result.update(
        status='near_closed_limit' if at_limit else 'above_closed_limit',
        joint_name=name,
        remaining_closing_travel_fraction=fraction,
        closed_limit_tolerance_fraction=tolerance,
        measurement_source='native_joint_pos_and_verified_physical_limits',
        guidance=(
            'The measured joint is at or near its closed physical limit, leaving negligible closing travel. '
            'Proximity and small relative motion may coexist with external support; they do not establish '
            'that this hand can retain the object. Inspect the visible grip before relying on independent support.'
            if at_limit else
            'The measured joint remains above its closed physical limit. This may reflect contact, '
            'obstruction or tracking lag; it is not a force measurement or grasp confirmation.'
        ),
    )
    return result
