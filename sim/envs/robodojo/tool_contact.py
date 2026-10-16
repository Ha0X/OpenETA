"""Explicit short tool strokes with actual contact bounds, never object plans.

Target motion is unknown until native physics runs. Planning excludes only the
declared tool/workpiece pairs; execution must refresh and monitor them at every
physics step. This is not a continuous dynamic collision certificate.
"""
import numpy as np

from .support_contact import SupportPlacement


def marked_dynamic_body(world, point):
    """Resolve a measured point privately without creating a finger grant."""
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Tool contact needs finite marked surface points')
    scene = world.scene
    probe = scene.fcl.CollisionObject(scene.fcl.Sphere(.003), scene.fcl.Transform(point))
    matches = []
    for root, indices in world.dynamic.items():
        for index in indices:
            hit = scene.fcl.CollisionResult()
            scene.fcl.collide(probe, scene.objects[index], scene.fcl.CollisionRequest(), hit)
            distance = 0. if hit.is_collision else scene.fcl.distance(
                probe, scene.objects[index], scene.fcl.DistanceRequest(), scene.fcl.DistanceResult())
            if distance <= .008:
                matches.append((root, tuple(indices)))
                break
    if len(matches) != 1:
        reason = 'no_body_match' if not matches else 'ambiguous_body_match'
        raise ValueError(f'Tool contact mark must identify exactly one dynamic body: {reason}, '
                         f'matched_body_count={len(matches)}. Select an exposed surface pixel away from contact boundaries.')
    return matches[0]


class ToolStroke(SupportPlacement):
    """Keep table support bounded while permitting one declared contact stroke."""
    max_stroke_m = .05

    def __init__(self, control, role, point, target_points, targets, *, patch_radius_m=.20, target_handles=(), mode='stroke'):
        from .checked_control import pose_error
        if mode not in ('stroke', 'reorient', 'adjust'):
            raise ValueError('Unknown tool contact mode')
        self.mode = mode
        if set(targets) != {role} or len(targets[role]) != 1:
            raise ValueError('Tool contact requires one arm and one short target')
        current = control.tcp(control.roles.index(role))
        target = np.asarray(targets[role][0], dtype=float)
        delta = target[:3, 3] - current[:3, 3]
        if mode == 'stroke' and (np.linalg.norm(delta) > self.max_stroke_m or abs(delta[2]) > .003 or
                pose_error(current, target)['orientation_rad'] > .001):
            raise ValueError('Tool stroke requires at most 5cm translation, 3mm height change and unchanged orientation')
        if mode != 'stroke' and (np.linalg.norm(delta) > .03 or
                pose_error(current, target)['orientation_rad'] > np.deg2rad(15)):
            raise ValueError('Tool contact adjustment requires at most 3cm TCP translation and 15 degrees rotation')
        if (not isinstance(target_handles, (list, tuple)) or
                any(not isinstance(h, str) or not h for h in target_handles) or
                not 0 <= len(target_points) + len(target_handles) <= 8):
            raise ValueError('Tool contact accepts at most eight fresh marks and prior workpiece references in total')
        # Reuse grasp, calibrated tabletop and local held-geometry validation.
        super().__init__(control, role, point, {role: [current]},patch_radius_m=patch_radius_m)
        self.requested = target.copy()
        self.effective = target.copy()  # no target clipping or semantic repair
        from .scene_geometry import forward_kinematics
        from .x5_calibration import link_to_tcp
        self.reference_tcp = link_to_tcp(forward_kinematics(
            control.world.scene.snapshot['robots'][self.index],self.positions[self.index])['link6'])
        from .motion_corridor import OrientationArc
        self.orientation_arc = OrientationArc(current, target)
        self.stroke_delta = delta.copy()
        held_root = control.contact_grants[role]['root']
        self.workpieces = {}
        self.workpiece_sources = {}
        self.source_action_index = getattr(getattr(control, 'adapter', None), '_step_count', None)
        self._public_references = None
        from .workpiece_references import WorkpieceReferences
        if not hasattr(control, 'workpiece_references'):
            control.workpiece_references = WorkpieceReferences()
        for mark_index, p in enumerate(target_points):
            root, indices = marked_dynamic_body(control.world, p)
            if root == held_root:
                raise ValueError('Tool workpiece must differ from the held tool')
            # A stationary receiving object is eligible only if explicitly
            # marked; no automatic permission for the opposite gripper/arm.
            self.workpieces[root] = indices
            self.workpiece_sources.setdefault(root, {'target_mark_indices': [], 'target_handles': []})['target_mark_indices'].append(mark_index)
        for token in target_handles:
            root, indices = control.workpiece_references.resolve(control, role, token)
            if root == held_root:
                raise ValueError('Tool workpiece must differ from the held tool')
            self.workpieces[root] = indices
            self.workpiece_sources.setdefault(root, {'target_mark_indices': [], 'target_handles': []})['target_handles'].append(token)
        self.workpiece_indices = tuple(sorted({i for inds in self.workpieces.values() for i in inds}))
        self.contact_pairs = [(control.world.scene.items[i]['id'], control.world.scene.items[j]['id'])
                              for i in self.indices for j in self.workpiece_indices]
        # Explicitly marked workpieces may contact each other (e.g. a receiver)
        # and the marked table. Unmarked world bodies are never added here.
        groups = list(self.workpieces.values())
        self.contact_pairs.extend((control.world.scene.items[i]['id'], control.world.scene.items[j]['id'])
            for k, group in enumerate(groups) for other in groups[k+1:] for i in group for j in other)
        self.contact_pairs.extend((control.world.scene.items[i]['id'],control.world.table_id)
                                  for i in self.workpiece_indices)
        self.table_pairs = list(self.pairs)
        self.pairs = [*self.table_pairs, *self.contact_pairs]
        self.last_workpiece_contact = None
        self._validate_workpiece_patch(control.world.scene, control.world.positions())
        initial = self.check(control.world.scene, control.world.positions(), measured=True)
        if initial['status'] != 'clear':
            raise ValueError('Existing tool contact exceeds measured contact bounds')

    def _validate_workpiece_patch(self, scene, positions):
        from .scene_geometry import forward_kinematics
        frames = [forward_kinematics(r, q) for r, q in zip(scene.snapshot['robots'], positions)]
        for index in self.workpiece_indices:
            item = scene.items[index]
            transform = (scene.world_transforms[index] if item['robot_index'] is None
                         else frames[item['robot_index']][item['body']])
            points = scene.vertices[index] @ transform[:3, :3].T + transform[:3, 3]
            if np.max(np.linalg.norm(points[:, :2] - self.point[:2], axis=1)) > self.patch_radius_m:
                raise ValueError('Marked workpiece leaves the tool contact patch')

    def check(self, scene, positions, *, measured=False):
        # During SupportPlacement construction, workpiece fields do not yet exist.
        if not hasattr(self, 'table_pairs'):
            return super().check(scene, positions, measured=measured)
        all_pairs = self.pairs
        try:
            self.pairs = self.table_pairs
            result = super().check(scene, positions, measured=measured)
        finally:
            self.pairs = all_pairs
        if result['status'] != 'clear':
            return result
        from .scene_geometry import forward_kinematics
        from .x5_calibration import link_to_tcp
        from .checked_control import pose_error
        tcp = link_to_tcp(forward_kinematics(scene.snapshot['robots'][self.index],positions[self.index])['link6'])
        displacement = tcp[:3,3]-self.reference_tcp[:3,3]
        alpha = np.clip(float(displacement@self.stroke_delta)/max(float(self.stroke_delta@self.stroke_delta),1e-12),0.,1.)
        path_error = float(np.linalg.norm(displacement-alpha*self.stroke_delta))
        orientation_error = (self.orientation_arc.distance(tcp) if self.mode != 'stroke'
                             else pose_error(tcp,self.reference_tcp)['orientation_rad'])
        if path_error>.003 or orientation_error>.01:
            return {'status':'blocked','reason':'tool_stroke_path_envelope_exceeded',
                    'contact_mode': self.mode,
                    'orientation_reference': 'shortest_commanded_arc' if self.mode != 'stroke' else 'initial_pose',
                    'evaluation':'measured_geometry' if measured else 'predicted_geometry',
                    'path_error_m':path_error,'path_limit_m':.003,
                    'orientation_error_rad':orientation_error,'orientation_limit_rad':.01,
                    'segment_progress_fraction':float(alpha)}
        try:
            self._validate_workpiece_patch(scene, positions)
        except ValueError:
            return {'status': 'blocked', 'reason': 'tool_workpiece_patch_exceeded'}
        # A frozen target does not predict a push. Only measured transforms are
        # evidence for tool/workpiece penetration; unknown target motion is
        # explicitly excluded from the planning guarantee.
        if measured and self.contact_pairs:
            distances = scene.signed_distances(positions, self.contact_pairs,
                                              contact_bounds=True,separation_bounds=True)
            if scene.last_contact_distance_method=='conservative_projection_bound_on_fcl_disagreement':
                self.bound_fallbacks+=1
            penetration = max(0., -min(distances.values()))
            self.last_workpiece_contact = {'maximum_penetration_m': penetration,
                                          'evaluation': 'measured_geometry',
                                          'penetration_limit_m': self.max_penetration_m,
                                          'distance_method':scene.last_contact_distance_method,
                                          'separated_pair_bounds':sum(m=='aabb_separation_lower_bound'
                                              for m in scene.last_contact_distance_methods.values()),
                                          'positive_distance_policy':'exact_or_conservative_separation_lower_bound; never_contact_evidence',
                                          **self._workpiece_pair_feedback(scene, distances)}
            if penetration > self.max_penetration_m:
                return {'status': 'blocked', 'reason': 'tool_workpiece_penetration_limit',
                        'measurement': self.last_workpiece_contact}
        return result

    def _workpiece_pair_feedback(self, scene, distances):
        from .checked_control import CheckedController
        ids = {item['id']: i for i, item in enumerate(scene.items)}
        methods = getattr(scene, 'last_contact_distance_methods', {})

        def describe(pair, distance):
            a, b = sorted(pair, key=ids.get)
            method = methods.get(pair, scene.last_contact_distance_method)
            return {'geometry_a': f'geometry_{ids[a]}', 'geometry_b': f'geometry_{ids[b]}',
                'participants': [CheckedController._part(self.control, ids[a]),
                                 CheckedController._part(self.control, ids[b])],
                'signed_clearance_m': distance, 'maximum_penetration_m': max(0., -distance),
                'distance_method': method,
                'penetration_estimate': ('no_overlap_certified_by_enclosing_boxes' if method=='aabb_separation_lower_bound'
                    else 'conservative_upper_bound' if method != 'fcl_signed_distance' else 'signed_geometry_distance')}

        ordered = sorted(distances.items(), key=lambda entry: (entry[1], sorted(entry[0])))
        limiting = [describe(pair, distance) for pair, distance in ordered[:4]]
        # Group only already-authorized identities and already-computed values.
        # Many convex table contacts must not hide a tool/workpiece near-pair.
        groups = {scene.items[i]['id']: ('held_tool', 0) for i in self.indices}
        for reference_index, indices in enumerate(self.workpieces.values()):
            groups.update({scene.items[i]['id']: ('workpiece', reference_index) for i in indices})
        groups[self.control.world.table_id] = ('table', 0)
        interactions, supports, seen = [], [], set()
        for pair, distance in ordered:
            if distance > .002:
                break  # Do not expose a general object-distance query.
            if any(identity not in groups for identity in pair):
                continue
            group_pair = tuple(sorted(groups[identity] for identity in pair))
            if group_pair in seen:
                continue
            seen.add(group_pair)
            kinds = {kind for kind, _ in group_pair}
            rows = supports if 'table' in kinds else interactions
            if len(rows) >= 4:
                continue
            row = describe(pair, distance)
            row['pair_scope'] = ('workpiece_to_table' if 'table' in kinds else
                                 'held_tool_to_workpiece' if 'held_tool' in kinds else
                                 'workpiece_to_workpiece')
            row['workpiece_reference_indices'] = [index for kind, index in group_pair if kind == 'workpiece']
            if 'held_tool' in kinds:
                row['tool_role'] = self.role
            rows.append(row)
        return {'limiting_pairs': limiting, 'checked_pair_count': len(distances),
                'near_interaction_pairs': interactions, 'near_workpiece_table_pairs': supports,
                'near_pair_band_m': .002, 'near_pair_limit_per_group': 4,
                'near_pair_basis': 'smallest computed distance per authorized body pair; conservative bounds may overstate proximity',
                'force_or_actual_contact_inferred': False,
                'guidance': 'Distances use the collision model at measured transforms. A conservative upper bound exceeding the limit is not proof of that much physical penetration; inspect the reported pairs before changing the contact plan.'}

    def reached(self):
        return False  # touch is intended during a stroke, not its terminal condition

    def public(self):
        # Publish only after constructor validation and request processing have
        # produced a receipt. Merely attempting an invalid selection issues no
        # partly constructed set of references.
        if self._public_references is None:
            self._public_references = []
            for root, indices in self.workpieces.items():
                try:
                    reference = self.control.workpiece_references.issue(
                        self.control, self.role, root, indices, self.source_action_index)
                except ValueError:
                    # A native stage may retire geometry after the move. Do
                    # not turn an already-executed action into an error merely
                    # because its optional future reference cannot be issued.
                    reference = {'status': 'unavailable', 'reason': 'identity_or_grasp_authorization_changed',
                                 'guidance': 'Use a fresh visible workpiece mark for later contact'}
                self._public_references.append({**reference, 'selected_by': self.workpiece_sources[root]})
        return {'mode': 'tool_'+self.mode, 'role': self.role,
                'scope': 'held_tool_and_explicitly_marked_workpieces_mutual_contact_and_table_patch_only',
                'workpiece_count': len(self.workpieces),
                'workpiece_references': self._public_references,
                'requested_tcp_xyz': self.requested[:3, 3].tolist(),
                'effective_tcp_xyz': self.effective[:3, 3].tolist(),
                'requested_tcp_rotation': self.requested[:3, :3].tolist(),
                'effective_tcp_rotation': self.effective[:3, :3].tolist(),
                'stop_on_contact': False, 'measurement': self.last,
                'workpiece_contact': self.last_workpiece_contact,
                'conservative_bound_evaluations':self.bound_fallbacks,
                'max_stroke_m': .03 if self.mode != 'stroke' else self.max_stroke_m,
                'max_rotation_rad': float(np.deg2rad(15)) if self.mode != 'stroke' else .001,
                'orientation_reference': 'shortest_commanded_arc' if self.mode != 'stroke' else 'initial_pose',
                'path_semantics': 'TCP line segment plus orientation envelope; no fixed object pivot or sustained support guarantee',
                'tcp_path_envelope_m':.003,'orientation_envelope_rad':.01,
                'guarantee': 'per_physics_step_measured_contact_bounds; target_motion_not_predicted',
                'authorization_lifetime': 'this_move_only', 'gripper_release': 'agent_decision'}
