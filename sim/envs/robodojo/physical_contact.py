"""Intent-free rigid finger contact; internal geometry is not tactile feedback.

Only real finger links may contact environment geometry. Arm/arm, self and
non-finger collisions keep existing swept checks. Workpiece dynamics are not
predicted and this policy provides no force or grasp certification.
"""
import math
import logging

POLICY = 'physical_contact_v1'
PENETRATION_LIMIT_M = .001


class PhysicalContact:
    def __init__(self, world, *, roles):
        self.world = world
        roles = tuple(roles)
        if (len(roles) != len(world.scene.snapshot['robots']) or len(set(roles)) != len(roles)
                or any(not isinstance(role, str) or not role for role in roles)):
            raise ValueError('Contact feedback requires one unique public role per robot')
        items = world.scene.items
        schemas = world.scene.snapshot['robots']
        fingers = {i: tuple(r.get('finger_links', ('link7', 'link8')))
                   for i, r in enumerate(schemas)}
        if any(len(names) != 2 or len(set(names)) != 2 for names in fingers.values()):
            raise ValueError('Physical contact requires two distinct verified finger links')
        self.finger_parts = {a['id']:{'kind':'robot', 'role':roles[a['robot_index']],
            'part':'finger_'+str(fingers[a['robot_index']].index(a['body'])+1)}
            for a in items if a['robot_index'] is not None
            and schemas[a['robot_index']].get('control_owner', 'target') != 'support'
            and a['body'] in fingers[a['robot_index']]}
        self.pairs = [(a['id'], b['id']) for a in items if a['id'] in self.finger_parts
                      for b in items if b['robot_index'] is None]
        self.has_auxiliary = any(r.get('control_owner') == 'support' for r in schemas)
        self.keys = {frozenset(p) for p in self.pairs}
        self.initial = {}
        self.bounds = {}
        self.samples = 0
        self.recovery = False

    def _depths(self, positions):
        scene = self.world.scene
        # Inspect only currently overlapping pairs. Broad-phase geometry and
        # bounded distances stay internal; no object identities reach the agent.
        check = scene.check(positions, allowed_pairs=self.world.support_pairs, clearance_limit=0.)
        pairs = [(c['a'], c['b']) for c in check['contacts']
                 if frozenset((c['a'], c['b'])) in self.keys]
        if not pairs:
            return {}
        distances = scene.signed_distances(positions, pairs, contact_bounds=True)
        if any(not math.isfinite(d) for d in distances.values()):
            raise ValueError('Contact penetration bound unavailable')
        return {frozenset(p):max(0., -float(d)) for p,d in distances.items()}

    def begin(self, positions):
        self.initial = self._depths(positions)
        self.bounds = {p:max(PENETRATION_LIMIT_M,d) for p,d in self.initial.items()}
        self.recovery = any(d > PENETRATION_LIMIT_M for d in self.initial.values())
        self.samples = 0

    def check(self, positions):
        try:
            depths = self._depths(positions)
        except (ValueError, RuntimeError):
            return {'clear':False, 'reason':'physical_contact_bound_unavailable',
                    'guidance':'Contact geometry could not be checked. No force or grasp inference is available.'}
        self.samples += 1
        exceeded = [pair for pair, depth in depths.items()
                    if depth > self.bounds.get(pair,PENETRATION_LIMIT_M)+1e-8]
        if exceeded:
            # Operator-only telemetry: never add private geometry or bounds to
            # the public receipt. It distinguishes actual solids from upper
            # bounds before anyone considers changing this safety policy.
            items = {item['id']:item for item in self.world.scene.items}
            methods = getattr(self.world.scene,'last_contact_distance_methods',{})
            logging.getLogger(__name__).warning('OPENETA_CONTACT_BOUND %s', [
                {'pair':sorted(pair), 'depth_bound_m':depths[pair],
                 'limit_m':self.bounds.get(pair,PENETRATION_LIMIT_M),
                 'initial_bound_m':self.initial.get(pair,0.),
                 'method':methods.get(pair,'unknown'),
                 'geometry_sources':[items[name].get('geometry_source','unspecified') for name in sorted(pair)]}
                for pair in exceeded])
            # Only public robot identity leaves the safety layer. Object names,
            # shape IDs, penetration values and grasp/force inferences do not.
            violations = {}
            for pair in exceeded:
                parts = sorted((self.finger_parts[name] for name in pair if name in self.finger_parts),
                               key=lambda part:(part['role'],part['part']))
                if len(parts) == 1:
                    parts.append({'kind':'scene_geometry'})
                key = tuple(tuple(part.items()) for part in parts)
                violations[key] = {'participants':parts,
                    'evidence':'discrete_geometry_bound_not_force_measurement'}
            ordered = sorted(violations.values(), key=lambda v:(v['participants'][0]['role'],v['participants'][0]['part']))
            return {'clear':False, 'reason':'physical_contact_penetration_limit',
                    'contact_bound_violations':ordered,
                    'guidance':'The finger/environment geometric bound was exceeded at the reported robot parts. '
                               'Inspect public images and choose a separating motion or gripper change. '
                               'The requested action role need not be the triggering role. '
                               'No automatic retreat, force measurement, or grasp inference was made.'}
        # A previously excessive contact may only improve or remain unchanged;
        # lost contact cannot subsequently reuse the larger initial bound.
        for pair in self.bounds:
            self.bounds[pair] = max(PENETRATION_LIMIT_M, min(self.bounds[pair],depths.get(pair,0.)))
        return {'clear':True}

    def public(self):
        return {'contact_policy':POLICY, 'intent_authorization_required':False,
                'contact_region':'physical_finger_links_to_environment',
                'penetration_limit_m':PENETRATION_LIMIT_M,
                'native_auxiliary_interactions': ({
                    'owner':'native_environment',
                    'delegated':'auxiliary_self_and_auxiliary_environment_interactions',
                    'host_checks':'target_robot_and_target_load_against_live_auxiliary_and_scene_geometry',
                    'dynamic_objects':'refreshed_each_physics_step; intervening_dynamics_not_predicted',
                    'auxiliary_grasp_inference':False,
                    'passive_finger_state':'unchanged_measured_auxiliary_finger_geometry_is_not_an_agent_joint_command; actual_geometry_retained; target_and_changed_joint_limits_enforced',
                    'stop_behavior':'pause_owned_simulation_and_preserve_native_queue'} if self.has_auxiliary else None),
                'initial_deep_contact_recovery':'nonworsening_only' if self.recovery else None,
                'measured_checks':self.samples, 'force_control':False,
                'grasp_inference':False, 'automatic_spatial_replanning':False,
                'held_object_prediction':False, 'held_object_world_sweep':'not_predicted',
                'coverage':'robot_self_and_nonfinger_world_sweep; finger_world_measured_bound',
                'collision_guarantee':'protected_robot_pairs_swept; finger_contact_discrete_measured_bounds; workpiece_dynamics_not_predicted'}
