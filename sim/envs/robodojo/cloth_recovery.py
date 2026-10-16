"""Bounded separation from an existing positive cloth protection gap.

This does not permit cloth penetration, new contact, or a guessed upward escape.
Only the existing finger pairs receive individual positive clearance floors.
"""
import numpy as np


def _distance(cloth, key):
    material, index = key
    scene = cloth.control.world.scene
    item = scene.items[index]
    role = scene.snapshot['robots'][item['robot_index']]['role']
    return cloth.snapshots[material].clearance(scene.objects[index], role=role, finger=True)


def certify_recovery(cloth, start, target, initial):
    rejected = {**initial, 'reason':'cloth_recovery_not_certified',
                'guidance':'Only small separating motions from positive finger gaps can recover. Other cloth contacts need a new strategy; no contact permission was expanded.'}
    contacts = initial['contacts']
    if (not contacts or any(c['role'] is None or c['body'] not in ('link7','link8')
                           or not 0 < c['clearance_lower_bound_m'] <= .0002 for c in contacts)):
        return rejected
    if len(getattr(cloth,'recovery_events',[])) >= 40:
        return {**rejected,'reason':'cloth_recovery_physics_budget'}
    scene = cloth.control.world.scene
    for item, chain in zip(scene.items, scene.relative_bounds.chains):
        index = item['robot_index']
        if index is None:
            continue
        names = scene.snapshot['robots'][index]['joint_names']
        bound = sum((r if kind=='revolute' else 1.)*abs(target[index][names.index(n)]-start[index][names.index(n)])
                    for n,kind,r in chain if kind!='fixed')
        if bound > .01:
            return {**rejected,'reason':'cloth_recovery_command_too_large','part_motion_limit_m':.01}
    before = {(c['cloth_index'],c['part_index']):c['clearance_lower_bound_m'] for c in contacts}
    floors = {}
    for (material,index),gap in before.items():
        floors.setdefault(material,{})[index] = max(0.,gap-.000002)
    cloth.check(target)  # Update all FK objects before exact pair queries.
    after = {key:_distance(cloth,key) for key in before}
    if any(after[key] <= gap+.0000005 for key,gap in before.items()):
        return {**rejected,'reason':'cloth_recovery_no_predicted_separation'}
    proof = cloth._certify_segment(start,target,floors=floors)
    if not proof['clear']:
        return {**rejected,'detail':proof}
    cloth.pending_recovery = before
    return {**proof,'reason':'bounded_positive_cloth_gap_separation',
            'normal_clearance_margin_restored':all(gap > .0002 for gap in after.values())}


def check_measured_recovery(cloth, positions, normal):
    before = cloth.pending_recovery
    cloth.pending_recovery = None
    # check() has just updated rigid objects against the NEW cloth snapshot.
    after = {key:_distance(cloth,key) for key in before}
    new = [c for c in normal['contacts'] if (c['cloth_index'],c['part_index']) not in before]
    worsened = any(not np.isfinite(after[k]) or after[k] <= 0 or after[k] < gap-1e-8
                   for k,gap in before.items())
    event = {'minimum_gap_before_m':min(before.values()),'minimum_gap_after_m':min(after.values()),
             'new_blocking_pairs':len(new),'measured_nonworsening':not worsened,
             'normal_margin_restored':normal['clear']}
    if not hasattr(cloth,'recovery_events'):
        cloth.recovery_events=[]
    cloth.recovery_events.append(event)
    if new or worsened:
        return {**normal,'clear':False,'reason':'cloth_recovery_measured_worsening_or_new_contact',
                'recovery':event}
    return {**normal,'clear':True,'recovery':event,
            'reason':'measured_positive_gap_recovery; normal_margin_not_yet_required_for_existing_pairs'}
