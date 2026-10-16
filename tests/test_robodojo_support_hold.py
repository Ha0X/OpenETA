import numpy as np
import pytest
from test_robodojo_support_contact import setup
from sim.envs.robodojo.support_contact import SupportPlacement
from sim.envs.robodojo.support_hold import SupportHold


def make_hold():
    c, targets = setup()
    p = SupportPlacement(c,'left',[-1,0,-.04],targets)
    with pytest.raises(ValueError,match='measured'):
        SupportHold.from_placement(p)
    t = np.eye(4);t[2,3] = -.0102
    c.world.scene.world_transforms[2] = t;c.world.scene.set_transform(2,t)
    p.check(c.world.scene,c.world.positions(),measured=True)
    return c,SupportHold.from_placement(p)


def test_stationary_hold_checks_support_and_grasp_and_owner_drift():
    c,h=make_hold()
    assert h.check(c.world.scene,c.world.positions(),measured=True)['status']=='clear'
    q=c.world.positions();q[0][0]+=.003
    assert h.check(c.world.scene,q)['reason']=='stationary_support_owner_moved'
    c.contact_grants={}
    assert h.check(c.world.scene,c.world.positions(),measured=True)['reason']=='stationary_support_authorization_changed'


def test_release_does_not_require_continued_grasp_but_keeps_support_bounds():
    c,h=make_hold();h.release_in_progress=True;c.contact_grants={}
    assert h.check(c.world.scene,c.world.positions(),measured=True)['status']=='clear'
    t=np.eye(4);t[2,3]=-.012
    c.world.scene.world_transforms[2]=t;c.world.scene.set_transform(2,t)
    assert h.check(c.world.scene,c.world.positions(),measured=True)['reason']=='support_penetration_limit'


def test_stationary_support_samples_ignore_unrelated_arm_but_keep_full_path_check():
    from types import SimpleNamespace
    from sim.envs.robodojo.support_contact import SupportContactScene
    c, hold = make_hold()
    checks = []
    snapshot = {'robots': [*c.world.scene.snapshot['robots'], c.world.scene.snapshot['robots'][0]]}
    scene = SimpleNamespace(snapshot=snapshot, items=c.world.scene.items,
        check_path=lambda start, target, **kw: checks.append((start, target)) or {'status': 'clear'})
    sampled = []
    hold.check = lambda s, q: sampled.append(q) or {'status': 'clear'}
    wrapped = SupportContactScene(scene, hold)
    start = [[0., 0., 0.], [0., 0., 0.]]
    target = [[0., 0., 0.], [1., 0., 0.]]
    result = wrapped.check_path(start, target)
    assert checks == [(start, target)]  # Full two-arm collision trajectory is still certified.
    assert result['support_samples'] == len(sampled) == 2
    assert result['support_sample_articulations'] == [0]
    target[0][0] = .01
    result = wrapped.check_path(start, target)
    assert result['support_samples'] == 6  # Owner motion still controls sampling resolution.
    scene.check_path = lambda *a, **kw: {'status': 'blocked', 'reason': 'fixture_collision'}
    sampled.clear()
    assert wrapped.check_path(start, target)['reason'] == 'fixture_collision'
    assert sampled == []


def test_support_sampling_includes_every_geometry_owner_and_unknown_subclasses():
    from copy import deepcopy
    from types import SimpleNamespace
    c, hold = make_hold()
    items = deepcopy(c.world.scene.items)
    scene = SimpleNamespace(items=items, snapshot={'robots': [object(), object()]})
    items[hold.indices[0]]['robot_index'] = 1
    assert hold.path_sampling_indices(scene) == [0, 1]
    items[hold.indices[0]]['robot_index'] = None
    items[hold.table]['robot_index'] = 1
    assert hold.path_sampling_indices(scene) == [0, 1]
    class ExtendedHold(SupportHold):
        pass
    extended = ExtendedHold.__new__(ExtendedHold)
    extended.__dict__.update(hold.__dict__)
    items[hold.table]['robot_index'] = None
    assert list(extended.path_sampling_indices(scene)) == [0, 1]
    assert list(SupportPlacement.path_sampling_indices(hold, scene)) == [0, 1]


def test_expiry_removes_only_selected_support_permission_and_reports_reason():
    from sim.envs.robodojo.checked_control import CheckedController
    c,h=make_hold()
    control=CheckedController.__new__(CheckedController)
    control.world=c.world;control.world.support_pairs=[]
    control.contact_grants=c.contact_grants
    control.support_holds={'left':h};control.support_events=[]
    control._rebuild_allowed_pairs()
    assert ('held','table') in control.allowed_pairs
    control._expire_support(['right'],'owner_move_requested')
    assert 'left' in control.support_holds and not control.support_events
    control._expire_support(['left'],'owner_move_requested')
    assert ('held','table') not in control.allowed_pairs
    assert ('finger7','held') in control.allowed_pairs
    assert control.support_events==[{'kind':'support_authorization_expired','role':'left',
                                    'reason':'owner_move_requested'}]


def test_invalid_contact_request_does_not_expire_existing_support():
    from sim.envs.robodojo.checked_control import CheckedController
    c,h=make_hold()
    control=CheckedController.__new__(CheckedController)
    control.world=c.world;control.world.support_pairs=[]
    control.contact_grants=c.contact_grants;control.roles=['left'];control.tcp=c.tcp
    control.support_holds={'left':h};control.support_events=[]
    control._rebuild_allowed_pairs();before=list(control.allowed_pairs)
    with pytest.raises(ValueError,match='radius'):
        control.move({'left':[c.tcp(0)]},support_point=[-1,0,-.04],patch_radius_m=.4)
    assert control.support_holds=={'left':h} and control.allowed_pairs==before
    assert control.support_events==[]


def test_current_evidence_reuse_never_persists_across_checks(monkeypatch):
    import sim.envs.robodojo.attached_geometry as geometry
    c,h=make_hold();calls=[]
    def query(*args):
        calls.append(1)
        return None,[]
    monkeypatch.setattr(geometry,'attached_scene',query)
    evidence=[{'role':'left','mode':'conditional_rigid_attachment'}]
    assert h.check(c.world.scene,c.world.positions(),measured=True,
                   holding_evidence=evidence)['status']=='clear'
    assert not calls
    assert h.check(c.world.scene,c.world.positions(),measured=True)['reason']== \
        'stationary_support_attachment_evidence_lost'
    assert len(calls)==1
    assert h.check(c.world.scene,c.world.positions(),measured=True,
                   holding_evidence=[])['reason']=='stationary_support_attachment_evidence_lost'
    c.contact_grants={}
    assert h.check(c.world.scene,c.world.positions(),measured=True,
                   holding_evidence=evidence)['reason']=='stationary_support_authorization_changed'


def test_guarded_step_requeries_support_after_physics(monkeypatch):
    from types import SimpleNamespace
    import sim.envs.robodojo.attached_geometry as geometry
    from sim.envs.robodojo.checked_control import CheckedController
    c,h=make_hold();control=CheckedController.__new__(CheckedController)
    control.world=c.world;control.contact_grants=c.contact_grants
    control.support_holds={'left':h};control.support_placement=None
    q=c.world.positions()[0]
    entity=SimpleNamespace(data=SimpleNamespace(joint_pos_target=q[None,:]))
    control.native=SimpleNamespace(robot_manager=SimpleNamespace(robot_key=[entity]))
    schema=c.world.scene.snapshot['robots'][0]
    indices=[schema['joint_names'].index(n) for n in schema['arm_joint_names']]
    control.arm_start=control.arm_targets=[q[indices]]
    control.operation='move';control.stopped=None;control.recovery=None
    control.attachment_baseline={};control.physics_steps=0
    control.attachment_monitor={'checked_physics_steps':0}
    control.allowed_pairs=[];events=[]
    evidence=[{'role':'left','mode':'conditional_rigid_attachment'}]
    def query(*args):
        events.append('query')
        return c.world.scene,evidence if control.physics_steps==0 else []
    monkeypatch.setattr(geometry,'attached_scene',query)
    monkeypatch.setattr(geometry,'attachment_frames',lambda *a:{})
    control._prediction_scene=lambda current:query()
    control._contact_scene=lambda scene:scene
    monkeypatch.setattr(c.world.scene,'check_path',lambda *a,**kw:{'status':'clear'})
    monkeypatch.setattr(c.world.scene,'check',lambda *a,**kw:{'clear':True})
    control._halt=lambda reason,detail=None:setattr(control,'stopped',{'reason':reason})
    control._guarded_step(lambda:events.append('physics'))
    assert events==['query','physics','query']
    assert control.stopped['reason']=='stationary_support_attachment_evidence_lost'
