from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.robodojo.attached_geometry import attached_scene
from sim.envs.robodojo.support_separation import SupportSeparation
from test_robodojo_support_contact import setup


def candidate(gap=-.0003, compound=False):
    c, targets=setup()
    if compound:
        from copy import deepcopy
        from sim.envs.robodojo.collision_geometry import CollisionScene
        snapshot=deepcopy(c.world.scene.snapshot)
        second=deepcopy(snapshot['colliders'][2]);second['id']='held_second'
        second['vertices']=(np.asarray(second['vertices'])+[0,0,.0004]).tolist()
        snapshot['colliders'].append(second);c.world.scene=CollisionScene(snapshot)
        c.contact_grants['left']['indices']=[2,4]
        c.contact_grants['left']['pairs'] += [('finger7','held_second'),('finger8','held_second')]
    t=np.eye(4);t[2,3]=gap-.01
    c.world.scene.world_transforms[2]=t
    c.world.scene.set_transform(2,t)
    if compound:
        c.world.scene.world_transforms[4]=t;c.world.scene.set_transform(4,t)
    c.native=SimpleNamespace(sim=SimpleNamespace(physics_dt=.004))
    targets['left'][0]=c.tcp(0);targets['left'][0][2,3]+=.02
    scene,evidence=attached_scene(c.world,c.world.positions(),c.contact_grants)
    contacts=scene.check(c.world.positions(),allowed_pairs=c.contact_grants['left']['pairs'])['contacts']
    return c,scene,contacts,evidence,targets


def test_positive_support_gap_uses_same_bounded_lift_off_without_changing_limit():
    c,scene,contacts,evidence,targets=candidate(gap=.00000012)
    assert not contacts
    r=SupportSeparation(c,scene,c.world.positions(),[{'a':'held','b':'table'}],evidence,targets)
    assert r.diagnostics()['initial_support_state']=='positive_near_gap'
    measured(r,[-.0000006,-.00101])
    assert r.check_actual(c.world.positions())
    assert not r.check_actual(c.world.positions())
    assert r.failure_reason=='support_separation_envelope_exceeded'
    c,scene,contacts,evidence,targets=candidate(gap=.0003)
    with pytest.raises(ValueError,match='within 0.2mm'):
        SupportSeparation(c,scene,c.world.positions(),[{'a':'held','b':'table'}],evidence,targets)


def test_compound_support_includes_nonpenetrating_parts_of_only_the_same_body():
    c,scene,contacts,evidence,targets=candidate(compound=True)
    assert not any('held_second' in (x['a'],x['b']) for x in contacts)
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    assert set(r.pairs)=={('held','table'),('held_second','table')}
    assert r.diagnostics()['support_geometry_part_count']==2
    readings=iter([[-.0003,.0001],[-.0002,-.00015],[-.0001,-.00005]])
    r.scene=SimpleNamespace(signed_distances=lambda *a:dict(zip(
        [frozenset(p) for p in r.pairs],next(readings))))
    # The nearest support part can change, while the whole body's minimum
    # clearance still improves. Independent per-part ratchets are unsuitable.
    assert r.check_segment([np.zeros(3)],[np.array([.004,0,0])])['status']=='clear'
    readings=iter([[-.0003,.0001],[-.0002,-.0004]])
    assert r.check_segment([np.zeros(3)],[np.array([.002,0,0])])['reason']=='existing_body_support_would_deepen'


def test_whole_body_actual_monitor_catches_new_part_penetration():
    c,scene,contacts,evidence,targets=candidate(compound=True)
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    r.actual_scene=SimpleNamespace(items=scene.items,signed_distances=lambda *a:{
        frozenset(('held','table')):.0001,frozenset(('held_second','table')):-.0011})
    assert not r.check_actual(c.world.positions())
    assert r.failure_reason=='support_separation_envelope_exceeded'


def measured(recovery, distances):
    values=iter(distances)
    recovery.actual_scene=SimpleNamespace(items=recovery.scene.items,
        signed_distances=lambda *args:{frozenset(('held','table')):next(values)})


def test_support_can_rebound_after_improving_without_exceeding_initial_envelope():
    c,scene,contacts,evidence,targets=candidate()
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    measured(r,[-.000255,-.000220,-.000233,.0003])
    assert all(r.check_actual(c.world.positions()) for _ in range(4))
    assert r.diagnostics()['reference']=='absolute_support_envelope_with_progress_and_time_limits'
    assert r.samples[-2]['maximum_deepening_from_best_m']>1e-5


def test_contact_transient_is_bounded_by_absolute_penetration_limit():
    c,scene,contacts,evidence,targets=candidate()
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    measured(r,[-.00038,-.00101])
    assert r.check_actual(c.world.positions())
    assert not r.check_actual(c.world.positions())
    assert r.failure_reason=='support_separation_envelope_exceeded'


def test_no_progress_cannot_wait_indefinitely_on_the_table():
    c,scene,contacts,evidence,targets=candidate()
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    measured(r,[-.0003]*r.progress_interval)
    assert all(r.check_actual(c.world.positions()) for _ in range(r.progress_interval-1))
    assert not r.check_actual(c.world.positions())
    assert r.failure_reason=='support_separation_no_measured_progress'


def test_slow_progress_does_not_extend_contact_time_forever():
    c,scene,contacts,evidence,targets=candidate()
    r=SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
    count=int(round(r.max_contact_s/r.period))
    measured(r,[-.0003+i*.000001 for i in range(1,count+1)])
    assert all(r.check_actual(c.world.positions()) for _ in range(count-1))
    assert not r.check_actual(c.world.positions())
    assert r.failure_reason=='support_separation_contact_time_limit'
    assert 'nonworsening_tolerance_m' not in r.last_check


def test_held_table_exception_never_applies_to_robot_or_unheld_object_contact():
    c,scene,contacts,evidence,targets=candidate()
    with pytest.raises(ValueError,match='independently held'):
        SupportSeparation(c,scene,c.world.positions(),contacts,[],targets)
    with pytest.raises(ValueError,match='only covers'):
        SupportSeparation(c,scene,c.world.positions(),[{'a':'finger7','b':'table'}],evidence,targets)
    targets['left'][0][0,3]+=.02
    with pytest.raises(ValueError,match='vertical lift'):
        SupportSeparation(c,scene,c.world.positions(),contacts,evidence,targets)
