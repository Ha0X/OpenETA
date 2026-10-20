"""Checked joint execution for the pinned fixed X5 profile.

The native action/reward loop remains the owner of physics and scoring. This
layer plans TCP knots, checks the chosen IK joint solution, and monitors every
native physical step. It never reports a discrete monitor as a continuous
dynamic collision guarantee.
"""
from contextlib import nullcontext
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .native_scene import NativeScene
from .native_fluid import FluidCoverageError
from .scene_geometry import array, pose_matrix, forward_kinematics
from .x5_calibration import link_to_tcp, tcp_to_link


def pose_error(actual, target):
    return {'position_m': float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            'orientation_rad': float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude())}


def cartesian_knots(start, target, position_step=.02, rotation_step=.10):
    error = pose_error(start, target)
    n = max(1, math.ceil(error['position_m']/position_step), math.ceil(error['orientation_rad']/rotation_step))
    if n > 100:
        raise ValueError('Cartesian request exceeds 100 bounded knots')
    rotations = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
    result = []
    for alpha in np.linspace(0, 1, n+1)[1:]:
        pose = np.eye(4)
        pose[:3, :3] = rotations(alpha).as_matrix()
        pose[:3, 3] = (1-alpha)*start[:3, 3]+alpha*target[:3, 3]
        result.append(pose)
    return result


class CheckedController:
    def __init__(self, adapter, *, allow_cloth=False, allow_articulations=False, allow_fluid=False, cloth_mode="cloth_short_horizon_v1"):
        self.adapter = adapter
        self.native = adapter._env
        if getattr(getattr(self.native.sim,'cfg',None),'decimation',None) != 1:
            raise ValueError('Per-physics-step monitoring requires native decimation=1')
        self.world = NativeScene(self.native, allow_cloth=allow_cloth, allow_articulations=allow_articulations, allow_fluid=allow_fluid)
        self.allowed_pairs = list(self.world.support_pairs)
        from .robot_ownership import RobotOwnership
        self.ownership = RobotOwnership.from_manager(self.native.robot_manager)
        self.ownership.require_target_prefix()
        self.roles = [self.world.scene.snapshot['robots'][i]['role'] for i in self.ownership.targets]
        self.support_execution = None
        if self.ownership.support:
            from .support_execution import SupportExecution
            self.support_execution = SupportExecution(self.native, self.ownership)
        self.receipt = None
        self.last_step = None
        self.stopped = None
        self.physics_steps = 0
        self.arm_targets = None
        self.arm_start = None
        self.contact_grants = {}
        self.support_holds = {}
        self.support_events = []
        self.recovery = None
        self.joint_step_limited_actions = 0
        from .ik_candidates import CuroboCandidates
        self.candidate_solver = CuroboCandidates(self.native, self.world.scene.snapshot)
        self.gripper_intents = self._grippers()
        self.operation = None
        self.attachment_baseline = {}
        self.physical_contact = None
        self.cloth = None
        if allow_cloth:
            from .cloth_execution import ClothExecution
            self.cloth = ClothExecution(self, mode=cloth_mode)
        self.fluid = None
        if allow_fluid and self.native.scene_manager._fluid_objects[0]:
            from .fluid_execution import FluidExecution
            self.fluid = FluidExecution(self)
            self.world._fluid_policy = self.fluid.policy

    def _active_contacts(self):
        return [*getattr(self,'support_holds',{}).values(), *([self.support_placement]
                if getattr(self,'support_placement',None) is not None else [])]

    def _rebuild_allowed_pairs(self):
        self.allowed_pairs = [*self.world.support_pairs,
            *getattr(getattr(self,'physical_contact',None),'pairs',[]),
            *(p for g in self.contact_grants.values() for p in g['pairs']),
            *(p for h in getattr(self,'support_holds',{}).values() for p in h.pairs)]

    def _expire_support(self, roles, reason):
        expired = False
        for role in roles:
            if getattr(self,'support_holds',{}).pop(role,None) is not None:
                expired = True
                self.support_events.append({'kind':'support_authorization_expired','role':role,'reason':reason})
        if expired:
            self._rebuild_allowed_pairs()

    def _contact_scene(self, scene):
        from .support_contact import SupportContactScene
        for contact in self._active_contacts():
            scene = SupportContactScene(scene,contact)
        return scene

    def _workpiece_collision_pairs(self):
        placement = getattr(self,'support_placement',None)
        return [(i,j) for indices in getattr(placement,'workpieces',{}).values()
                for i in indices for j,item in enumerate(self.world.scene.items)
                if item['robot_index'] is None and j not in indices]

    def authorize_contact(self, role, point_world, *, cloth_contact='surface', approach=None):
        if role not in self.roles:
            raise ValueError('Unknown contact effector')
        self.world.refresh()
        scene = self.world.scene
        point = array(point_world) + array(self.native.robot_manager.scene.env_origins)[0]
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid marked surface point')
        if getattr(self,'cloth',None) is not None:
            self.cloth.refresh()
            authorization = self.cloth.authorize(role,point,contact_mode=cloth_contact,approach=approach)
            if authorization is not None:
                return authorization
        fcl = scene.fcl
        probe = fcl.CollisionObject(fcl.Sphere(.003), fcl.Transform(point))
        matches = []
        for root, indices in self.world.dynamic.items():
            distances = []
            for i in indices:
                hit = fcl.CollisionResult()
                fcl.collide(probe, scene.objects[i], fcl.CollisionRequest(), hit)
                distances.append(0. if hit.is_collision else fcl.distance(probe, scene.objects[i],
                                 fcl.DistanceRequest(), fcl.DistanceResult()))
            if min(distances) <= .008:
                matches.append((root, indices))
        if len(matches) != 1:
            return {'authorized': False, 'reason': 'marked_surface_not_one_dynamic_contact_body',
                    'candidate_count': len(matches)}
        root, indices = matches[0]
        arm = self.roles.index(role)
        pairs = [(item['id'], scene.items[j]['id']) for item in scene.items
                 if item['robot_index'] == arm and item['body'] in ('link7', 'link8') for j in indices]
        self._expire_support([role],'contact_reauthorization')
        self.contact_grants[role] = {'root': root, 'indices': indices, 'pairs': pairs}
        self._rebuild_allowed_pairs()
        return {'authorized': True, 'role': role, 'scope': 'selected_body_to_selected_finger_links_only',
                'contact_id': f'contact_{min(indices)}', 'source': 'agent_marked_calibrated_rgbd_surface',
                'grasp_confirmed': False}

    def gripper(self, role, openness, *, max_actions=None):
        if role not in self.roles or not np.isfinite(openness) or not 0 <= openness <= 1:
            raise ValueError('Use a named gripper and openness in [0,1]')
        from .holding_motion import HoldingMotionWindow
        if max_actions is None:
            max_actions = 50 if openness <= .05 else 25
        # A 0.16s sample can label a slow, persistent rotation as small.
        # Full closure may need longer to settle; keep early exit and a hard cap.
        self.holding_motion = HoldingMotionWindow(float(self.native.sim.physics_dt), window_s=.4)
        self.operation = 'gripper'
        self.attachment_baseline = {}
        self.stopped = None
        self.physics_steps = 0
        self.last_step = None
        self.recovery = None
        self.joint_step_limited_actions = 0
        selected = self.roles.index(role)
        hold = getattr(self,'support_holds',{}).get(role)
        if hold is not None:
            hold.release_in_progress = openness > .95
        self.world.refresh()
        positions = self.world.positions()
        targets = [q[[r['joint_names'].index(n) for n in r['arm_joint_names']]]
                   for q, r in zip(positions, self.world.scene.snapshot['robots'])]
        grippers = self._grippers()
        grippers[selected] = float(openness)
        initial = self.world.scene.check(positions, allowed_pairs=self.allowed_pairs, clearance_limit=.002)
        if not initial['clear']:
            grant = self.contact_grants.get(role)
            ids = {item['id']: item for item in self.world.scene.items}
            body_ids = {self.world.scene.items[j]['id'] for j in grant['indices']} if grant else set()
            def releasable(contact):
                a, b = contact['a'], contact['b']
                robot = ids[b] if a in body_ids else ids[a] if b in body_ids else {}
                return robot.get('robot_index') == selected and robot.get('body') in ('link6', 'link7', 'link8')
            if openness > .95 and all(releasable(c) for c in initial['contacts']):
                from .boundary_recovery import BoundaryRecovery
                self.recovery = BoundaryRecovery(self.world.scene, positions, initial['contacts'], release=True)
            else:
                self.stopped = {'reason': 'gripper_initial_collision',
                    'detail': self._public_check({'detail': initial}),
                    'guidance': 'Only opening the authorized contacting hand permits bounded release; otherwise request a short retreat.'}
        if initial['clear'] and initial['clearance_m'] <= .0002:
            from .clearance_recovery import ClearanceRecovery, near_clearance_pairs
            near, _ = near_clearance_pairs(self.world.scene,positions,self.allowed_pairs)
            if near:
                self.recovery=ClearanceRecovery(self.world.scene,positions,near,allowed_pairs=self.allowed_pairs,
                    table_id=self.world.table_id)
        recovery_mode = getattr(self.recovery, 'mode', None)
        recovery_monitor = self.recovery
        previous = None
        stable = 0
        first_width_stable_action = None
        settling = {"enabled": openness <= .05, "status": "awaiting_stable_opening" if openness <= .05 else "not_requested",
            "minimum_window_s": .4, "translation_limit_m": .0005,
            "rotation_limit_rad": .01, "native_action_limit": max_actions,
            "grasp_confirmed": False}
        count = 0
        root = self.adapter._last_obs['proprio']
        grip = root.get('arms', {'primary': root})[role]['gripper_state']
        for count in range(1, max_actions+1):
            if self.stopped:
                count -= 1
                break
            self.gripper_intents[selected] = float(openness)
            self._action(targets[:len(self.roles)], grippers)
            root = self.adapter._last_obs['proprio']
            grip = root.get('arms', {'primary': root})[role]['gripper_state']
            actual = float(grip['openness'])
            stable = stable+1 if previous is not None and abs(actual-previous) < .002 else 0
            previous = actual
            ready = stable >= 3
            if ready and first_width_stable_action is None:
                first_width_stable_action = count
            if ready and settling['enabled']:
                recent = self.holding_motion.public()['roles'].get(role)
                if recent is None:
                    settling['status'] = 'no_current_attachment_evidence'
                else:
                    ready = (recent['window_sim_s'] >= settling['minimum_window_s'] and
                        recent['maximum_relative_translation_m'] <= settling['translation_limit_m'] and
                        recent['maximum_relative_rotation_rad'] <= settling['rotation_limit_rad'])
                    settling['status'] = 'relative_motion_small' if ready else 'waiting_for_relative_motion'
            if self.stopped or self.adapter._terminal or ready:
                break
        if settling['status'] in ('waiting_for_relative_motion','awaiting_stable_opening') and count>=max_actions:
            settling['status']='budget_exhausted'
        settling['extra_actions_after_width_stable'] = (max(0,count-first_width_stable_action)
            if first_width_stable_action is not None else 0)
        if openness > .95 and float(grip['openness']) > .95:
            self.contact_grants.pop(role, None)
            self._expire_support([role],'measured_release')
            self._rebuild_allowed_pairs()
        if hold is not None:
            hold.release_in_progress = False
        from .closure_diagnostics import closure_diagnostics
        return {'schema_version': 'openeta.robodojo_checked_gripper.v1', 'requested_openness': openness,
                'measured': grip, 'native_actions': count, 'physics_steps': self.physics_steps,
                'closure_diagnostics': closure_diagnostics(grip, openness),
                'stop': self.stopped, 'grasp_confirmed': False,
                'recovery_mode': recovery_mode,
                'recovery_diagnostics': recovery_monitor.diagnostics() if hasattr(recovery_monitor,'diagnostics') else None,
                'initial_clearance_boundary': ({**self._public_check({'detail':recovery_monitor.initial_clearance_check}),
                    'classification':'positive_gap_not_overlap'} if hasattr(recovery_monitor,'initial_clearance_check') else None),
                'host_adjustments': ([{'kind': 'positive_clearance_retreat' if recovery_mode=='positive_clearance_retreat' else 'bounded_release',
                                      'reason': 'positive_gap_below_normal_margin' if recovery_mode=='positive_clearance_retreat' else 'existing_authorized_hand_object_collision',
                                      'existing_contact_guarantee': 'swept_nonintersection_plus_sampled_nonshrinking_gap' if recovery_mode=='positive_clearance_retreat' else 'sampled_nonworsening_with_physics_monitor'}]
                                     if recovery_mode else []) + ([{'kind':'bounded_gripper_settling',
                                        'reason':'opening_stable_but_relative_body_motion_not_yet_small',
                                        **settling}] if settling['extra_actions_after_width_stable'] else []),
                'gripper_intent_latched': count > 0,
                'contact_authorization_retained': role in self.contact_grants,
                'holding_motion': self.holding_motion.public(),
                'gripper_settling': settling,
                'grasp_evidence': 'inspect_finger_state_relative_motion_and_visual_lift; closure_alone_is_insufficient',
                'collision_policy': self.world.public_policy()}

    def _public_plan(self, value):
        if isinstance(value, dict):
            if isinstance(value.get('detail'), dict) and 'contacts' in value['detail']:
                value = self._public_check(value)
            return {k: self._public_plan(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._public_plan(v) for v in value]
        return value

    def tcp(self, index):
        if index not in range(len(self.roles)):
            raise ValueError('TCP control is restricted to target robots')
        robot = self.native.robot_manager.robot_list[index]
        entity = self.native.robot_manager.robot_key[index]
        link = pose_matrix(array(entity.data.body_link_pose_w)[0, entity.body_names.index(robot.ee_link_name)])
        origin = array(self.native.robot_manager.scene.env_origins)[0]
        tcp = link_to_tcp(link)
        tcp[:3, 3] -= origin
        return tcp

    def _public_check(self, result):
        result = dict(result)
        detail = result.pop('detail', None)
        if detail and 'contacts' not in detail:
            # Recovery proofs can wrap a path verdict (including budget
            # exhaustion), rather than a sampled collision result. Preserve
            # its cause and recursively anonymize any nested geometry details.
            result['detail'] = self._public_plan(detail)
            detail = None
        if detail:
            ids = {item['id']: i for i, item in enumerate(self.world.scene.items)}
            result['contacts'] = [{'geometry_a': f'geometry_{ids[c["a"]]}',
                                   'geometry_b': f'geometry_{ids[c["b"]]}',
                                   'robot_a': c['robot_a'], 'robot_b': c['robot_b'],
                                   'participants': [self._part(ids[c['a']]), self._part(ids[c['b']])],
                                   **{key:c[key] for key in ('previous_signed_clearance_m','measured_signed_clearance_m') if key in c}}
                                  for c in detail['contacts']]
            result['clearance_lower_bound_m'] = detail['clearance_m']
            def public_pair(a,b):
                return {'geometry_a':f'geometry_{ids[a]}','geometry_b':f'geometry_{ids[b]}',
                        'participants':[self._part(ids[a]),self._part(ids[b])]}
            if detail.get('closest_pair'):
                result['closest_pair']=public_pair(*detail['closest_pair'])
            limiting=detail.get('uncertified_pairs',[])
            if limiting:
                result['clearance_limiting_pairs']=[public_pair(self.world.scene.items[i]['id'],
                    self.world.scene.items[j]['id']) for i,j in limiting[:8]]
                result['clearance_limiting_pair_count']=len(limiting)
            if result.get('reason')=='insufficient_clearance' and not detail['contacts']:
                result['guidance']='No overlap was reported at the sampled configuration, but the requested path clearance margin is not certified. Inspect the limiting parts and revise the approach or request a short separating retreat.'
            for key in ('maximum_deepening_m','minimum_vertex_height_change_m','nonworsening_tolerance_m',
                        'support_penetration_limit_m','reference','maximum_deepening_from_initial_m'):
                if key in detail:
                    result[key] = detail[key]
            if detail.get('constraint_violations'):
                result['constraint_violations'] = detail['constraint_violations']
        if result.get('status') == 'unknown':
            result.setdefault('guidance', 'No clearance proof within budget; shorten the motion or retreat into open space.')
        return result

    def _part(self, index):
        item = self.world.scene.items[index]
        if item['robot_index'] is not None:
            return {'kind':'robot','role':self.world.scene.snapshot['robots'][item['robot_index']]['role'],
                    'part':{'link7':'finger_1','link8':'finger_2'}.get(item['body'],item['body'])}
        if item['id']==self.world.table_id:
            return {'kind':'static_support','part':'table'}
        result={'kind':'scene_geometry','id':f'geometry_{index}'}
        authorizations=[{'role':role,'contact_id':f'contact_{min(grant["indices"])}'}
            for role,grant in getattr(self,'contact_grants',{}).items() if index in grant['indices']]
        if authorizations:
            result.update(contact_authorizations=authorizations,grasp_confirmed=False)
        return result

    def _solve_pose(self, index, pose, current):
        candidates, detail = self._pose_candidates(index, pose, current)
        if candidates:
            return candidates[0], None
        return None, {'reason': 'ik_search_no_valid_solution', 'status': 'unknown',
                      'role': self.roles[index], 'ik': detail}

    def _pose_candidates(self, index, pose, current):
        if index not in range(len(self.roles)):
            raise ValueError('IK commands are restricted to target robots')
        from .ik_candidates import validated_candidates
        candidates, detail = self.candidate_solver.solve(index, pose, current)
        valid, rejected = validated_candidates(self.world.scene.snapshot['robots'][index], current,
            candidates, pose, array(self.native.robot_manager.scene.env_origins)[0])
        return valid, {**detail, 'physical_rejections': rejected, 'valid_unique_candidates': len(valid)}

    def _halt(self, reason, detail=None, *, preserve_gripper_targets=False):
        if self.stopped is None:
            self.stopped = {'reason': reason, 'detail': detail, 'after_physics_steps': self.physics_steps}
        # Clear only this owned native queue. No reset of episode/rewards and
        # no extra simulation step: the next accepted command overwrites hold.
        support = getattr(self, 'support_execution', None)
        if support is not None:
            feedback = support.pause()
            if feedback is not None:
                self.stopped['support_execution'] = feedback
        manager = self.native.robot_manager.control_manager
        for queue in manager.control_queue:
            while not queue.is_empty():
                queue.pop()
        hold = {'arm_policy':'measured_position', 'joint_velocity_targets':'zero',
                'preserve_gripper_position_targets_requested':bool(preserve_gripper_targets),
                'snapshot_timing':'zero_physics_step_hold_after_last_action_observation',
                'extra_physics_steps':0, 'grasp_or_force_confirmed':False, 'grippers':[]}
        from .robot_ownership import RobotOwnership
        owners = RobotOwnership.from_manager(self.native.robot_manager)
        owners.require_target_prefix()
        for index in owners.targets:
            entity = self.native.robot_manager.robot_key[index]
            target = entity.data.joint_pos.clone()
            robot = self.native.robot_manager.robot_list[index]
            record = None
            if robot.ee_type == 'gripper':
                record = {'role':self.roles[index], 'policy':'measured_position',
                          'readback_status':'unavailable'}
                try:
                    slots = list(robot.gripper_joint_indices)
                    previous = array(entity.root_physx_view.get_dof_position_targets())
                    measured = array(entity.data.joint_pos)
                    if (len(slots) != 2 or len(set(slots)) != 2
                            or previous.shape != measured.shape
                            or not np.isfinite(previous[:,slots]).all()):
                        raise ValueError('Invalid prior finger drive targets')
                    record.update(previous_position_targets=previous[0,slots].tolist(),
                                  measured_joint_positions=measured[0,slots].tolist())
                    if preserve_gripper_targets:
                        target[:,slots] = target.new_tensor(previous[:,slots])
                        record['policy'] = 'previous_position_targets'
                    record['requested_hold_position_targets'] = array(target)[0,slots].tolist()
                except Exception:
                    # Optional command-state evidence must never prevent the
                    # measured-position safety hold or retain an invalid target.
                    target = entity.data.joint_pos.clone()
                    record.update(policy='measured_position_fallback',reason='drive_target_snapshot_unavailable')
            entity.set_joint_position_target(target)
            entity.set_joint_velocity_target(entity.data.joint_vel.clone()*0)
            entity.write_data_to_sim()
            if record is not None:
                if 'requested_hold_position_targets' in record:
                    try:
                        actual = array(entity.root_physx_view.get_dof_position_targets())[0,slots]
                        if np.isfinite(actual).all():
                            record.update(readback_status='available',position_targets_after_hold=actual.tolist(),
                                readback_matches_requested=bool(np.allclose(actual,record['requested_hold_position_targets'],atol=1e-7,rtol=0)))
                    except Exception:
                        pass
                before = record.get('previous_position_targets')
                after = record.get('position_targets_after_hold')
                record['drive_targets_changed'] = (bool(not np.allclose(before,after,atol=1e-7,rtol=0))
                    if before is not None and after is not None else None)
                record['guidance'] = (
                    'Previous finger position targets were retained. This does not confirm grasp or force.'
                    if record['policy']=='previous_position_targets' else
                    'Safety hold replaced finger drive targets with measured positions. This can reduce '
                    'closing effort, including on an unselected arm. Holding an object is not guaranteed; '
                    'inspect public observations before another transport. No extra physics step was taken. '
                    'The last requested gripper command is unchanged. Unless superseded by a new gripper '
                    'command, a later accepted motion or wait reapplies it through the normal safety checks.')
                hold['grippers'].append(record)
        self.stopped['actuator_hold'] = hold

    def _guarded_step(self, original, *args, **kwargs):
        if self.stopped:
            return
        horizon = getattr(self, '_execution_horizon', None)
        if horizon is not None and not horizon.before_step(self):
            return
        try:
            self.world.refresh()
            current = self.world.positions()
            if getattr(self,'physical_contact',None) is not None:
                contact_check = self.physical_contact.check(current)
                if not contact_check['clear']:
                    self._halt(contact_check['reason'], contact_check)
                    return
            targets = []
            for i, (entity, robot) in enumerate(zip(self.native.robot_manager.robot_key,
                                                   self.world.scene.snapshot['robots'])):
                desired = array(entity.data.joint_pos_target)[0].copy()
                indices = [robot['joint_names'].index(n) for n in robot['arm_joint_names']]
                if i < len(self.roles):
                    delta = self.arm_targets[i]-self.arm_start[i]
                    length2 = float(delta @ delta)
                    alpha = float((desired[indices]-self.arm_start[i]) @ delta/length2) if length2 > 1e-12 else 0.
                    on_segment = self.arm_start[i]+np.clip(alpha, 0, 1)*delta
                    if not np.allclose(desired[indices], on_segment, atol=2e-5):
                        self._halt('native_joint_target_differs_from_checked_command')
                        return
                # Task-owned targets remain in the joint collision check below;
                # only the Agent-command ownership assertion excludes them.
                limits = array(robot['joint_limits'])
                # Native gripper overtravel is explicit in measured feedback.
                # Collision kinematics use the physically reachable position.
                desired = np.clip(desired, limits[:, 0], limits[:, 1])
                targets.append(desired)
            if getattr(self,'cloth',None) is not None:
                self.cloth.refresh()
                cloth_check=self.cloth.check_segment(current,targets)
                if not cloth_check['clear']:
                    self._halt('before_physics_cloth_check',cloth_check)
                    return
            if getattr(self,'fluid',None) is not None:
                self.fluid.refresh()
                fluid_check=self.fluid.check_segment(current,targets)
                if not fluid_check['clear']:
                    self._halt('before_physics_fluid_check',fluid_check)
                    return
            recovery_pairs = self.recovery.pairs if self.recovery else []
            from .attached_geometry import attachment_frames
            prediction, attachments = self._prediction_scene(current)
            measured_attachment = attachment_frames(self.world, current, self.contact_grants, attachments)
            if self.operation == 'move' and set(self.attachment_baseline)-set(measured_attachment):
                self.attachment_monitor['status'] = 'evidence_lost'
                self._halt('conditional_attachment_evidence_lost', {
                    'roles': sorted(set(self.attachment_baseline)-set(measured_attachment)),
                    'guidance': 'Both-finger support is no longer observed; inspect and regrasp before transport.'})
                return
            for role, frame in measured_attachment.items():
                self.attachment_baseline.setdefault(role, frame)
            if self.operation == 'move':
                self.attachment_monitor['roles_with_evidence'] = sorted(measured_attachment)
                self.attachment_monitor['checked_physics_steps'] += 1
            placement=getattr(self,'support_placement',None)
            for contact in self._active_contacts():
                from .support_hold import SupportHold
                evidence = ({'holding_evidence': attachments}
                            if self.operation == 'move' and isinstance(contact, SupportHold) else {})
                checked_contact=contact.check(self.world.scene,current,measured=True,**evidence)
                if checked_contact['status']!='clear':
                    self._halt(checked_contact['reason'],checked_contact)
                    return
            workpiece_pairs=self._workpiece_collision_pairs()
            if workpiece_pairs:
                contact_pairs=placement.pairs if placement else []
                actual=self.world.scene.check(current,
                    allowed_pairs=[*self.allowed_pairs,*recovery_pairs,*contact_pairs],
                    extra_pairs=workpiece_pairs,clearance_limit=.02)
                if not actual['clear']:
                    self._halt('unpermitted_workpiece_contact_before_physics',self._public_check({'detail':actual}))
                    return
            prediction=self._contact_scene(prediction)
            protected = getattr(self, '_support_protected_pairs', set())
            execution_pairs = [p for p in [*self.allowed_pairs,*recovery_pairs] if frozenset(p) not in protected]
            check = prediction.check_path(current, targets, allowed_pairs=execution_pairs,
                                                timeout_s=4, max_nodes=1024)
            if check['status'] != 'clear':
                self._halt('before_physics_collision_check', self._public_check(check))
                return
            if self.recovery:
                if hasattr(self.recovery, 'check_segment'):
                    check = self.recovery.check_segment(current, targets)
                else:
                    from .separating_recovery import certify_upward
                    check = certify_upward(self.world.scene,current,targets,self.recovery.indices)
                if check['status'] != 'clear':
                    self._halt('separating_recovery_check',self._public_check(check))
                    return
            support = getattr(self, 'support_execution', None)
            if support is not None:
                support.begin_physics()
            original(*args, **kwargs)
            if support is not None:
                support.commit_physics()
            self.physics_steps += 1
            if horizon is not None:
                horizon.after_step()
            self.world.refresh()
            if getattr(self,'physical_contact',None) is not None:
                contact_check = self.physical_contact.check(self.world.positions())
                if not contact_check['clear']:
                    self._halt(contact_check['reason'], contact_check)
            if getattr(self,'cloth',None) is not None:
                self.cloth.refresh(advanced=True)
                cloth_check=self.cloth.check_after_physics(self.world.positions())
                if not cloth_check['clear']:
                    self._halt('measured_cloth_clearance_after_physics',cloth_check)
                self.cloth.measure()
            if getattr(self,'fluid',None) is not None:
                self.fluid.refresh()
                fluid_check=self.fluid.check(self.world.positions(),deadline=time.monotonic()+2.)
                if not fluid_check['clear']:
                    self._halt('after_physics_fluid_check',self.fluid.feedback(fluid_check))
            extra_pairs = []
            for attached in attachments:
                indices = self.contact_grants[attached['role']]['indices']
                extra_pairs.extend((i, j) for i in indices for j, item in enumerate(self.world.scene.items)
                                   if item['robot_index'] is None and j not in indices)
            extra_pairs.extend(workpiece_pairs)
            support_pairs=placement.pairs if placement else []
            check = self.world.scene.check(self.world.positions(), allowed_pairs=[p for p in [*self.allowed_pairs,*recovery_pairs,*support_pairs] if frozenset(p) not in protected],
                                           clearance_limit=.02, extra_pairs=extra_pairs)
            if not check['clear']:
                self._halt('measured_collision_after_physics', self._public_check({'detail': check}))
            if self.operation == 'gripper' and getattr(self,'holding_motion',None) is not None:
                # Measurement only: opening may legitimately remove evidence.
                # No extra physics, continued-motion approval, or grasp oracle.
                from .attached_geometry import attached_scene
                measured_positions=self.world.positions()
                _, live_evidence=attached_scene(self.world,measured_positions,self.contact_grants)
                frames=attachment_frames(self.world,measured_positions,self.contact_grants,live_evidence)
                self.holding_motion.record(frames)
            after_attachment = attachment_frames(self.world, self.world.positions(), self.contact_grants, attachments)
            for role, frame in after_attachment.items():
                error = pose_error(frame, self.attachment_baseline[role])
                monitor = self.attachment_monitor['relative_motion'].setdefault(role,
                    {'maximum_position_m':0., 'maximum_orientation_rad':0.})
                monitor['maximum_position_m'] = max(monitor['maximum_position_m'], error['position_m'])
                monitor['maximum_orientation_rad'] = max(monitor['maximum_orientation_rad'], error['orientation_rad'])
                if error['position_m'] > .004 or error['orientation_rad'] > .05:
                    self.attachment_monitor['status'] = 'relative_motion_exceeded'
                    self._halt('conditional_attachment_changed', {'role': role, 'relative_pose_change': error,
                        'guidance': "The object changed pose relative to the hand, invalidating this motion's rigid-attachment assumption. Motion stopped; this is not proof of a drop. Inspect the images. If still supported, the same closed-gripper command can measure recent relative motion without moving the wrists; persistent motion or lost support calls for placement/regrasp. Do not simply split the same failed transport into repeated moves."})
            if self.recovery and not self.recovery.check_actual(self.world.positions()):
                detail = self._public_check({'detail': self.recovery.last_check})
                reason = getattr(self.recovery,'failure_reason',None) or 'measured_existing_contact_deepened'
                detail['guidance'] = {
                    'measured_positive_gap_not_preserved':'The initially positive near-pair gap shrank beyond the measured tolerance or reached zero. Motion stopped; inspect the reported parts and choose a different separating direction.',
                    'support_separation_envelope_exceeded':'Held-body/table penetration exceeded the 1mm support limit. Inspect the contact and regrasp; bounded opening remains available.',
                    'support_separation_no_measured_progress':'The held body did not make sufficient progress away from the table. Inspect whether the grasp actually supports the body and regrasp.',
                    'support_separation_contact_time_limit':'The held body remained at the table beyond the bounded lift-off interval. Inspect the grasp and contact before a new attempt.'
                }.get(reason,'The measured contact worsened during physics. Inspect the contacting parts; opening the authorized hand remains available when bounded release checks pass.')
                self._halt(reason, detail)
            for contact in self._active_contacts():
                support=contact.check(self.world.scene,self.world.positions(),measured=True)
                if support['status']!='clear':
                    self._halt(support['reason'],support)
                elif contact is placement and placement.reached():
                    self._halt('support_contact_reached',placement.public())
            if self.recovery:
                normal = self.world.scene.check(self.world.positions(),allowed_pairs=self.allowed_pairs,clearance_limit=.001)
                distances = self.world.scene.signed_distances(self.world.positions(), self.recovery.pairs)
                if normal['clear'] and normal['clearance_m'] > .0002 and min(distances.values()) > .0002:
                    self.recovery = None
        except FluidCoverageError:
            self._halt('fluid_state_unavailable', {
                'guidance':'Fluid safety data is unavailable. Motion stopped without an automatic retreat. No liquid amount, force, or task result was inferred.'})
        except Exception:
            # An unexpected exception after dispatch stays an unknown adapter
            # outcome. Never manufacture a recoverable official step result.
            self._halt('physics_monitor_exception')
            raise

    def _prediction_scene(self, current):
        self._support_protected_pairs = set()
        if getattr(self, 'support_execution', None) is not None:
            from .support_geometry import protected_actor_pairs
            from .attached_geometry import attached_scene
            # Native auxiliary fingers may push, release, or grasp. Do not infer
            # a rigid attachment from proximity. Refresh world bodies at every
            # physical step; their intervening dynamics remain unpredicted.
            scene, evidence = attached_scene(self.world, current, self.contact_grants)
            self._support_protected_pairs = protected_actor_pairs(scene)
            if self.recovery is not None:
                self.recovery.scene = scene
            return scene, evidence if self.operation == 'move' else []
        # Recovery changes which existing contacts may separate, not whether
        # an authorised body must still follow the measured fingers/wrist.
        if self.operation == 'move':
            from .attached_geometry import attached_scene
            scene, evidence = attached_scene(self.world, current, self.contact_grants)
            if self.recovery is not None:
                self.recovery.scene = scene
            return scene, evidence
        return (self.recovery.scene if self.recovery is not None else self.world.scene), []

    def _action(self, arm_targets, grippers):
        if len(arm_targets) != len(self.roles) or len(grippers) != len(self.roles):
            raise ValueError('Native commands require exactly the configured target effectors')
        self.arm_start = [q[[r['joint_names'].index(n) for n in r['arm_joint_names']]]
                          for q, r in zip(self.world.positions(), self.world.scene.snapshot['robots'])][:len(self.roles)]
        from .motion_rate import joint_step_limits, record_rate
        limits = joint_step_limits(self, grippers)
        arm_targets = [array(q) for q in arm_targets]
        record_rate(self, limits, self.arm_start, arm_targets)
        self.arm_targets = []
        scale = min([1.] + [limit/max(float(abs(array(target)-start).max()), 1e-12)
                            for limit, start, target in zip(limits, self.arm_start, arm_targets)])
        limited = scale < 1.
        for start,target in zip(self.arm_start,arm_targets):
            delta = array(target)-start
            self.arm_targets.append(start+scale*delta)
        self.joint_step_limited_actions += int(limited)
        action = {'arms': {role: {'mode': 'joint_position', 'positions': q.tolist(), 'gripper': g}
                           for role, q, g in zip(self.roles, self.arm_targets, grippers)}}
        had_override = 'sim_step' in vars(self.native)
        previous = vars(self.native).get('sim_step')
        original = self.native.sim_step
        self.native.sim_step = lambda *a, **kw: self._guarded_step(original, *a, **kw)
        try:
            from .gripper_continuity import GripperTargetContinuity
            if not hasattr(self, 'gripper_continuity'):
                self.gripper_continuity = GripperTargetContinuity(self.native, self.roles)
            from .arm_continuity import ArmTargetContinuity
            support = getattr(self, 'support_execution', None)
            with ArmTargetContinuity(self, grippers).action(), self.gripper_continuity.action(grippers), (support.action() if support is not None else nullcontext()):
                self.last_step = self.adapter.step(action)
        except Exception:
            if getattr(self, 'support_execution', None) is not None:
                self.adapter._state_valid = False
            raise
        finally:
            if had_override:
                self.native.sim_step = previous
            else:
                del self.native.sim_step

    def _grippers(self):
        if hasattr(self, 'gripper_intents'):
            return list(self.gripper_intents)
        proprio = self.adapter._last_obs['proprio']
        arms = proprio.get('arms', {'primary': proprio})
        return [float(arms[role]['gripper_command']) for role in self.roles]

    def move(self, targets, *, timeout_s=360, max_actions=None, coordination='synchronized', support_point=None, retain_support=False, tool_points=None, patch_radius_m=.20, tool_handles=(), tool_mode='stroke', push_contact=None):
        from .motion_planning import validate_motion_targets
        validate_motion_targets(self,targets)
        if push_contact is not None:
            if support_point is not None or tool_points is not None or tool_handles or retain_support or coordination!='synchronized':
                raise ValueError('Push contact requires a separate single-arm contact request')
            from .push_contact import execute_push
            return execute_push(self,targets,push_contact,timeout_s=timeout_s,max_actions=max_actions)
        if tool_mode not in ('stroke', 'reorient', 'adjust') or (tool_mode != 'stroke' and (support_point is None or tool_points is None)):
            raise ValueError('Tool reorientation requires explicit table contact authorization')
        if coordination not in ('independent','synchronized'):
            raise ValueError('Unknown motion coordination semantics')
        if tool_handles and tool_points is None:
            raise ValueError('Workpiece references require an explicit tool contact stroke')
        self.attachment_monitor = {'status':'conditional_evidence_only', 'grasp_confirmed':False,
            'reference_scope':'this_move', 'checked_physics_steps':0,
            'roles_with_evidence':[], 'relative_motion':{}}
        self.support_placement=None
        if support_point is not None:
            from .support_contact import SupportPlacement
            if coordination!='synchronized' or len(targets)!=1:
                raise ValueError('Support placement requires one independent single-arm request')
            role=next(iter(targets))
            if tool_points is not None:
                if retain_support:
                    raise ValueError('Tool stroke cannot retain stationary support')
                from .tool_contact import ToolStroke
                placement=ToolStroke(self,role,support_point,tool_points,targets,patch_radius_m=patch_radius_m,
                                     target_handles=tool_handles, mode=tool_mode)
            else:
                placement=SupportPlacement(self,role,support_point,targets,patch_radius_m=patch_radius_m)
            self._expire_support(targets,'owner_move_requested')
            self.support_placement=placement
            try:
                from .motion_planning import execute_request
                if placement.reached():
                    receipt={'schema_version':'openeta.robodojo_checked_motion.v1','reached_target':False,
                             'stop':{'reason':'support_contact_reached','physics_advanced':False},
                             'native_actions':0,'physics_steps':0,'host_adjustments':[]}
                else:
                    receipt=execute_request(self,{role:[placement.effective]},timeout_s=timeout_s,max_actions=max_actions)
                field='tool_contact' if tool_points is not None else 'support_contact'
                receipt[field]=placement.public()
                if tool_points is not None:
                    receipt['collision_guarantee']=placement.public()['guarantee']
                    if 'planning' in receipt:
                        receipt['planning']['collision_guarantee']='static_unmarked_geometry_only; marked_workpiece_motion_unknown'
                        receipt['planning']['contact_path_envelope_m']=.003
                        receipt['planning']['contact_orientation_envelope_rad']=.01
                kind = {'stroke':'bounded_tool_stroke','reorient':'bounded_tool_reorientation',
                        'adjust':'bounded_tool_pose_adjustment'}
                receipt['host_adjustments'].append({'kind':kind[tool_mode] if tool_points is not None else 'bounded_support_placement',**placement.public()})
                if retain_support and placement.reached() and (receipt.get('stop') or {}).get('reason') in (None,'support_contact_reached'):
                    from .support_hold import SupportHold
                    hold=SupportHold.from_placement(placement)
                    self.support_holds[role]=hold
                    self._rebuild_allowed_pairs()
                    self.support_events.append({'kind':'support_authorization_retained',**hold.public()})
                return receipt
            finally:
                self.support_placement=None
        self._expire_support(targets,'owner_move_requested')
        if coordination == 'independent':
            from .coordinated_motion import execute_independent
            return execute_independent(self, targets, timeout_s=timeout_s, max_actions=max_actions)
        if coordination != 'synchronized':
            raise ValueError('Unknown motion coordination semantics')
        from .motion_planning import execute_request
        return execute_request(self, targets, timeout_s=timeout_s, max_actions=max_actions)

    def _recovery_move(self, targets, *, timeout_s=120, max_actions=180):
        """targets: role -> list of TCP homogeneous transforms, world origin."""
        from .motion_precision import tolerance
        position_tolerance = tolerance(getattr(self, 'position_tolerance_m', .003))
        # Recovery remains a short bounded operation, even with adaptive normal motion budgets.
        max_actions = min(180, max_actions) if max_actions is not None else 180
        from .motion_budget import execution_budget, exhausted
        budget=execution_budget(0,max_actions,timeout_s)
        budget['policy']='bounded_contact_recovery'
        budget['required_nominal_actions']=None
        if not targets or not set(targets) <= set(self.roles):
            raise ValueError('Unknown or missing effector roles')
        self.operation = 'move'
        self.attachment_baseline = {}
        if any(not 1 <= len(path) <= 5 for path in targets.values()):
            raise ValueError('Supply one destination and at most four waypoints')
        self.stopped = None
        self.physics_steps = 0
        self.last_step = None
        self.recovery = None
        self.joint_step_limited_actions = 0
        started = time.monotonic()
        recovery_used = False
        plans = {}
        for role, poses in targets.items():
            current = self.tcp(self.roles.index(role))
            plans[role] = []
            for target in poses:
                target = array(target)
                if (target.shape != (4,4) or not np.isfinite(target).all()
                        or not np.allclose(target[3], [0,0,0,1])
                        or not np.allclose(target[:3,:3].T @ target[:3,:3], np.eye(3), atol=1e-6)
                        or not np.isclose(np.linalg.det(target[:3,:3]),1,atol=1e-6)):
                    raise ValueError('TCP target must be a finite rigid transform')
                plans[role].extend(cartesian_knots(current, target))
                current = target
        actions = 0
        completed = 0
        total = max(map(len, plans.values()))
        initial = [self.tcp(i) for i in range(len(self.roles))]
        held_joints = [q[[r['joint_names'].index(n) for n in r['arm_joint_names']]]
                       for q,r in zip(self.world.positions(),self.world.scene.snapshot['robots'])]
        self.world.refresh()
        from .attached_geometry import attached_scene
        recovery_scene, recovery_attachments = attached_scene(self.world, self.world.positions(), self.contact_grants)
        initial_check = recovery_scene.check(self.world.positions(),allowed_pairs=self.allowed_pairs,clearance_limit=.02)
        margin_recovery=None
        if initial_check['clear'] and initial_check['clearance_m'] <= .0002:
            from .clearance_recovery import ClearanceRecovery, near_clearance_pairs
            near, _ = near_clearance_pairs(recovery_scene,self.world.positions(),self.allowed_pairs)
            if near:
                margin_recovery=ClearanceRecovery(recovery_scene,self.world.positions(),near,allowed_pairs=self.allowed_pairs,
                    table_id=self.world.table_id)
                margin_recovery.actual_scene=self.world.scene
        if not initial_check['clear'] or margin_recovery is not None:
            short = all(len(path)==1 and pose_error(initial[self.roles.index(role)],array(path[0]))['position_m']<=.05
                        and pose_error(initial[self.roles.index(role)],array(path[0]))['orientation_rad']<=.025
                        for role,path in targets.items())
            if not short:
                self.stopped = {'reason':'clearance_recovery_requires_short_translation' if margin_recovery else 'contact_recovery_requires_short_translation',
                    'guidance':'Use a single retreat of at most 5 cm with the current orientation. Initially positive gaps must not shrink; other pairs retain their normal clearance checks.' if margin_recovery else 'Use a retreat of at most 5 cm with the current orientation. Finger/table recovery must lift every touching finger vertex.'}
            elif margin_recovery is not None:
                # A supported compound body can alternate between a tiny gap
                # and compliant contact. Use the same bounded lift-off rule,
                # only for this authorized held body and calibrated table.
                from .support_separation import SupportSeparation
                try:
                    self.recovery=SupportSeparation(self,recovery_scene,self.world.positions(),
                        [{'a':a,'b':b} for a,b in margin_recovery.pairs],recovery_attachments,targets)
                except ValueError:
                    self.recovery=margin_recovery
                recovery_used=True
            else:
                from .separating_recovery import TableSeparation
                try:
                    self.recovery = TableSeparation(recovery_scene,self.world.positions(),initial_check['contacts'],table=self.world.table_id)
                    recovery_used = True
                except ValueError as exc:
                    from .boundary_recovery import BoundaryRecovery
                    try:
                        from .support_separation import SupportSeparation
                        try:
                            self.recovery = SupportSeparation(self,recovery_scene,self.world.positions(),
                                initial_check['contacts'],recovery_attachments,targets)
                        except ValueError:
                            self.recovery = BoundaryRecovery(recovery_scene, self.world.positions(), initial_check['contacts'])
                            self.recovery.actual_scene = self.world.scene
                        recovery_used = True
                    except ValueError:
                        self.stopped = {'reason':'unsupported_contact_recovery','guidance':str(exc)}
        if self.recovery is not None and not initial_check['clear']:
            from .clearance_recovery import include_near_clearance
            self.recovery = include_near_clearance(self.recovery, recovery_scene,
                self.world.positions(), allowed_pairs=self.allowed_pairs,
                table_id=self.world.table_id, actual_scene=self.world.scene)
        # Check every requested endpoint before moving towards any of them.
        # This is kinematic admission, not a substitute for segment checks.
        for role,path in targets.items():
            if self.stopped:
                break
            i = self.roles.index(role)
            for endpoint,pose in enumerate(path):
                _, failure = self._solve_pose(i,array(pose),self.world.positions()[i])
                if failure:
                    self.stopped = {**failure,'phase':'endpoint_preview','endpoint':endpoint,
                                    'physics_advanced':False}
                    break
            if self.stopped:
                break
        recovery_mode = getattr(self.recovery, 'mode', 'monotone_finger_table_separation') if recovery_used else None
        recovery_monitor = self.recovery
        while completed < total and not self.stopped:
            poses = [plans[role][min(completed, len(plans[role])-1)] if role in plans else initial[i]
                     for i, role in enumerate(self.roles)]
            stop=exhausted(budget,started,actions,time.monotonic())
            if stop:
                self.stopped = stop
                break
            current = self.world.positions()
            joint_targets = []
            for i, (pose, robot, schema) in enumerate(zip(poses, self.native.robot_manager.robot_list,
                                                        self.world.scene.snapshot['robots'])):
                if self.roles[i] not in targets:
                    joint_targets.append(held_joints[i])
                    continue
                solved, failure = self._solve_pose(i,pose,current[i])
                if failure:
                    self.stopped = {**failure,'phase':'path_knot','knot':completed}
                    break
                joint_targets.append(solved)
            if self.stopped:
                break
            self._action(joint_targets, self._grippers())
            actions += 1
            if self.adapter._terminal:
                self.stopped = {'reason': 'official_episode_ended'}
                break
            errors = [pose_error(self.tcp(i), pose) for i, pose in enumerate(poses)]
            active_tolerance = min(.002, position_tolerance) if completed == total-1 else .002
            if all(e['position_m'] < active_tolerance and e['orientation_rad'] < .025 for e in errors):
                completed += 1
        final = {role: pose_error(self.tcp(self.roles.index(role)), array(path[-1])) for role, path in targets.items()}
        self.receipt = {'schema_version': 'openeta.robodojo_checked_motion.v1',
            'reached_target': completed == total and self.stopped is None,
            'stop': self.stopped, 'completed_knots': completed, 'planned_knots': total,
            'native_actions': actions, 'physics_steps': self.physics_steps,
            'joint_step_limit_rad': .04, 'recovery_joint_step_limit_rad': .02,
            'joint_step_limited_actions': self.joint_step_limited_actions,
            'ik_equivalence_policy':'nearest_full_turn_equivalent_within_physical_joint_limits',
            'recovery_mode': recovery_mode,
            'initial_collision': self._public_check({'detail': initial_check}) if not initial_check['clear'] else None,
            'conditional_attachments': recovery_attachments,
            'recovery_diagnostics': recovery_monitor.diagnostics() if hasattr(recovery_monitor,'diagnostics') else None,
                'initial_clearance_boundary': ({**self._public_check({'detail':recovery_monitor.initial_clearance_check}),
                    'classification':'positive_gap_not_overlap'} if hasattr(recovery_monitor,'initial_clearance_check') else None),
            'host_adjustments': ([{'kind':'contact_recovery','mode':recovery_mode,
                                  'scope':'existing_contacts_keep_original_recovery; positive_near_pairs_keep_swept_nonintersection; other_pairs_keep_normal_margin' if recovery_mode=='mixed_contact_and_positive_clearance_retreat' else 'initially_positive_below_margin_pairs_only; other_pairs_keep_normal_margin' if recovery_mode=='positive_clearance_retreat' else 'all_convex_parts_of_one_authorized_held_body_to_table; other_pairs_keep_swept_checks' if recovery_mode=='bounded_support_separation' else 'existing_contacts_only; other_pairs_keep_swept_checks',
                                  **({'initial_support_state':recovery_monitor.initial_support_state,
                                      'near_support_admission_m':.0002,'support_penetration_limit_m':.001}
                                     if recovery_mode=='bounded_support_separation' else {})}] if recovery_mode else []),
            'existing_contact_guarantee': ('original_contact_recovery_plus_separate_positive_gap_certificate' if recovery_mode=='mixed_contact_and_positive_clearance_retreat' else 'swept_nonintersection_plus_sampled_nonshrinking_gap' if recovery_mode=='positive_clearance_retreat' else 'predicted_nonworsening_with_bounded_actual_support_contact' if recovery_mode=='bounded_support_separation'
                else 'discrete_nonworsening_samples' if recovery_mode in ('sampled_nonworsening_retreat', 'bounded_release') else 'monotone_vertex_height' if recovery_mode else None),
            'unselected_effectors': 'held_at_initial_measured_joint_targets',
            'elapsed_s': time.monotonic()-started, 'final_tcp_error': final,
            'budget':budget,
            'execution': 'bounded_steps_toward_checked_ik_joint_targets', 'collision_policy': self.world.public_policy(),
            'collision_guarantee': 'stationary_world_joint_segment_certificate_with_discrete_physics_monitor'}
        return self.receipt
