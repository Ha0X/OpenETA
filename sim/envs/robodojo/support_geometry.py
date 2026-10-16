"""Ownership boundary for private collision geometry, never task observations.

Native auxiliary manipulation is delegated to the environment. Geometry stays
live for target/auxiliary checks; proximity never creates an auxiliary grasp.
"""


def native_interaction(snapshot, a, b):
    owners = [r.get('control_owner', 'target') for r in snapshot['robots']]
    indices = [item['robot_index'] for item in (a, b) if item['robot_index'] is not None]
    return bool(indices) and all(owners[i] == 'support' for i in indices)


def protected_actor_pairs(scene):
    """Finger/world permissions cannot permit contact with another robot actor."""
    owners = [r.get('control_owner', 'target') for r in scene.snapshot['robots']]
    return {frozenset((a['id'],b['id'])) for a in scene.items if a['robot_index'] is not None
            for b in scene.items if b['robot_index'] is not None
            and owners[a['robot_index']] != owners[b['robot_index']]}


def stationary_auxiliary_finger_slots(robot, start, target):
    """Measured passive geometry is not an Agent command to exceed a limit.

    Only unchanged, verified prismatic finger coordinates of native actors
    qualify. Their actual values remain in FK and swept collision bounds.
    """
    if robot.get('control_owner') != 'support':
        return set()
    from .collision_state import stationary_measured_finger_slots
    return stationary_measured_finger_slots(robot, start, target)
