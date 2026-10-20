import json

import pytest

from test_robodojo_tool_contact import stroke_setup
from sim.envs.robodojo.tool_contact import ToolStroke
from sim.envs.robodojo.workpiece_references import WorkpieceReferences


def selected(c, targets):
    return ToolStroke(c, 'left', [-1, 0, -.04], [[-1, .09, -.02]], targets)


def test_reference_reuses_identity_without_new_visible_point(monkeypatch):
    c, targets = stroke_setup()
    first = selected(c, targets).public()['workpiece_references'][0]
    token = first['handle']
    assert first['selected_by']['target_mark_indices'] == [0]
    monkeypatch.setattr('sim.envs.robodojo.tool_contact.marked_dynamic_body',
                        lambda *a: pytest.fail('A reference must not guess a new hidden pixel'))
    repeated = ToolStroke(c, 'left', [-1, 0, -.04], [], targets, target_handles=[token])
    assert repeated.workpieces == {'workpiece': (4,)}
    assert repeated.pairs == [('held', 'table'), ('held', 'workpiece'), ('workpiece', 'table')]
    public = repeated.public()['workpiece_references'][0]
    assert public['handle'] == token
    assert public['selected_by'] == {'target_mark_indices': [], 'target_handles': [token]}
    assert repeated.public()['authorization_lifetime'] == 'this_move_only'


def test_public_reference_contains_no_private_directory_or_pose():
    c, targets = stroke_setup()
    reference = selected(c, targets).public()['workpiece_references'][0]
    assert set(reference) == {'handle', 'source_action_index', 'scope', 'meaning', 'selected_by'}
    json.dumps(reference)  # no private grant dictionary or geometry objects
    assert reference['handle'].startswith('workpiece_')
    assert reference['scope'] == 'same_episode_role_and_grasp_authorization'


@pytest.mark.parametrize('change', ['release', 'reauthorize_same_body', 'another_role', 'another_episode'])
def test_reference_cannot_cross_grasp_role_or_episode(change):
    c, targets = stroke_setup()
    token = selected(c, targets).public()['workpiece_references'][0]['handle']
    ledger = c.workpiece_references
    role = 'left'
    if change == 'release':
        c.contact_grants.pop(role)
    elif change == 'reauthorize_same_body':
        c.contact_grants[role] = dict(c.contact_grants[role])
    elif change == 'another_role':
        c.contact_grants['right'] = c.contact_grants['left']
        role = 'right'
    else:
        c, _ = stroke_setup()
        ledger = WorkpieceReferences()
    with pytest.raises(ValueError):
        ledger.resolve(c, role, token)


@pytest.mark.parametrize('change', ['removed', 'indices', 'geometry_id'])
def test_live_geometry_directory_is_checked(change):
    c, targets = stroke_setup()
    token = selected(c, targets).public()['workpiece_references'][0]['handle']
    if change == 'removed':
        del c.world.dynamic['workpiece']
    elif change == 'indices':
        c.world.dynamic['workpiece'] = [2]
    else:
        c.world.scene.items[4]['id'] = 'replacement'
    with pytest.raises(ValueError):
        c.workpiece_references.resolve(c, 'left', token)


def test_reference_does_not_bypass_live_contact_patch():
    import numpy as np
    c, targets = stroke_setup()
    token = selected(c, targets).public()['workpiece_references'][0]['handle']
    transform = np.eye(4); transform[1, 3] = .5
    c.world.scene.world_transforms[4] = transform
    c.world.scene.set_transform(4, transform)
    with pytest.raises(ValueError, match='patch'):
        ToolStroke(c, 'left', [-1, 0, -.04], [], targets, target_handles=[token])


def test_reference_does_not_authorize_fingers_or_other_bodies():
    import numpy as np
    from sim.envs.robodojo.support_contact import SupportContactScene
    c, targets = stroke_setup()
    token = selected(c, targets).public()['workpiece_references'][0]['handle']
    p = ToolStroke(c, 'left', [-1, 0, -.04], [], targets, target_handles=[token])
    transform = np.eye(4); transform[1, 3] = -.04
    c.world.scene.world_transforms[4] = transform
    c.world.scene.set_transform(4, transform)
    result = SupportContactScene(c.world.scene, p).check(
        c.world.positions(), allowed_pairs=c.contact_grants['left']['pairs'])
    assert not result['clear']
    assert any('finger7' in (x['a'], x['b']) for x in result['contacts'])


def test_invalid_selection_issues_no_partial_references(monkeypatch):
    c, targets = stroke_setup()
    monkeypatch.setattr(WorkpieceReferences, 'issue', lambda *a: pytest.fail('Invalid selection published a reference'))
    with pytest.raises(ValueError, match='exactly one'):
        ToolStroke(c, 'left', [-1, 0, -.04], [[-1, .09, -.02], [-1, .5, 0]], targets)


def test_combined_selection_limit_and_unknown_tokens():
    c, targets = stroke_setup()
    token = selected(c, targets).public()['workpiece_references'][0]['handle']
    with pytest.raises(ValueError, match='eight'):
        ToolStroke(c, 'left', [-1, 0, -.04], [[-1, .09, -.02]], targets, target_handles=[token]*8)
    with pytest.raises(ValueError, match='Unknown workpiece reference'):
        ToolStroke(c, 'left', [-1, 0, -.04], [], targets, target_handles=['workpiece_guessed'])


def test_repeated_marks_reuse_one_identity_and_report_each_input():
    c, targets = stroke_setup()
    first = selected(c, targets).public()['workpiece_references'][0]['handle']
    p = ToolStroke(c, 'left', [-1, 0, -.04], [[-1, .09, -.02]]*2, targets, target_handles=[first])
    refs = p.public()['workpiece_references']
    assert len(refs) == p.public()['workpiece_count'] == 1
    assert refs[0]['handle'] == first
    assert refs[0]['selected_by'] == {'target_mark_indices': [0, 1], 'target_handles': [first]}


def test_geometry_retired_after_motion_does_not_hide_motion_receipt():
    c, targets = stroke_setup()
    p = selected(c, targets)
    del c.world.dynamic['workpiece']
    r = p.public()
    assert r['workpiece_references'][0]['status'] == 'unavailable'
    assert 'handle' not in r['workpiece_references'][0]
    assert r['mode'] == 'tool_stroke'
