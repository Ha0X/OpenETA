"""Config-bound atomic RoboDojo Host; LIBERO keeps its existing Host."""
from __future__ import annotations

import json
import hashlib
import re
import threading
import time
from pathlib import Path

import jsonschema
import numpy as np
from mcp.types import Tool, TextContent, ImageContent

from tools.harness_transport import StreamableHttpSimulatorMcpTransport
from tools.harness_robodojo import official_outcome, camera_directory
from tools.harness_evidence import CallEvidence, write_once
from tools.robodojo_skill_library import BUNDLES, SkillLibrary


class RemoteExecutionUnknown(RuntimeError):
    """The server explicitly invalidated its post-action state."""


def motion_feedback(receipt):
    """Summarize existing sanitized check evidence; never inspect scene state."""
    if not isinstance(receipt, dict) or receipt.get('reached_target'):
        return None
    roots = [receipt.get('stop')]
    coordination = receipt.get('coordination') or {}
    if coordination.get('status') == 'rejected':
        roots.append(coordination)
    pairs, limiting, positive_gaps, contact_bounds, contexts, reasons = [], [], [], [], [], set()
    def visit(value):
        if isinstance(value, dict):
            if isinstance(value.get('reason'), str):
                reasons.add(value['reason'])
            context = value.get('collision_evidence')
            if isinstance(context, dict):
                context = {k:v for k,v in context.items() if k in ('selection','candidate_rank','scope')}
                if context and context not in contexts:contexts.append(context)
            for field, result in [('contacts', pairs), ('clearance_limiting_pairs', limiting),
                                  ('contact_bound_violations', contact_bounds)]:
                for contact in value.get(field, []) or []:
                    parts = contact.get('participants', [])
                    if len(parts) == 2:
                        public = [{k:v for k,v in part.items() if k in ('kind','role','part')}
                                  for part in parts]
                        destination = result
                        gap = contact.get('measured_signed_clearance_m')
                        if field == 'contacts' and isinstance(gap,(int,float)) and gap > 0:
                            destination = positive_gaps
                        if public not in destination:
                            destination.append(public)
            for key, child in value.items():
                if key not in ('contacts','clearance_limiting_pairs','contact_bound_violations','collision_evidence'):
                    visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(roots)
    budget = bool(reasons & {'clearance_certificate_budget','cartesian_preview_budget_exhausted',
                             'candidate_generation_budget'})
    if not pairs and not budget and not positive_gaps and not contact_bounds:
        return None
    kinds = set()
    for a,b in pairs:
        if a.get('kind') == b.get('kind') == 'robot':
            kinds.add('self_collision' if a.get('role') and a.get('role') == b.get('role')
                      else 'inter_arm_collision')
        else:
            kinds.add('robot_scene_collision' if 'robot' in (a.get('kind'), b.get('kind'))
                      else 'scene_collision')
    preview = (receipt.get('native_actions') == 0 and receipt.get('physics_steps') == 0)
    guidance = []
    if 'self_collision' in kinds:
        guidance.append('Some tested configurations collide within the same arm. '
                        'These self-collision pairs concern one arm; evaluate any other reported collision kinds separately. '
                        "Consider the requested orientation and that arm's posture as well as position.")
    if 'inter_arm_collision' in kinds:
        guidance.append('Reported pairs include different robot arms; inspect their relative configurations.')
    if 'robot_scene_collision' in kinds:
        guidance.append('Reported pairs include robot and scene geometry; inspect the existing detailed evidence.')
    if positive_gaps:
        guidance.append('Some tracked gaps shrank while remaining positive. No overlap was reported for '
                        'those positive-gap pairs; evaluate any other reported collisions separately.')
    if contact_bounds:
        guidance.append('Finger/environment geometric contact bounds were exceeded at the reported '
                        'robot parts. They may belong to a different arm from the requested action. '
                        'This is not a force measurement or grasp confirmation.')
    if contexts:
        guidance.append('Each IK collision detail describes its labeled representative tested candidate, '
                        'not every rejected configuration. All tested candidates still require clearance.')
    if budget:
        guidance.append('Part of the clearance check was not certified within its budget. '
                        'Limiting pairs are not measured collision contacts; finite search does not prove infeasibility.')
    return {'source':'summary_of_existing_public_receipt',
            'evidence_scope':'tested_planning_configurations_no_physics' if preview else 'reported_safety_checks',
            'reported_collision_kinds':sorted(kinds), 'reported_collision_pairs':pairs[:8],
            'positive_gap_regression_pairs':positive_gaps[:8],
            'contact_bound_violation_pairs':contact_bounds[:8], 'collision_evidence_contexts':contexts[:8],
            'certificate_budget_exhausted':budget, 'clearance_limiting_pairs':limiting[:8],
            'guidance':guidance, 'scene_state_read':False, 'target_changed':False}


def object_schema(properties=None, required=()):
    return {'type': 'object', 'properties': properties or {}, 'required': list(required), 'additionalProperties': False}


def vector_schema(n=3):
    return {'type': 'array', 'items': {'type': 'number'}, 'minItems': n, 'maxItems': n}


def target_matrix(value, current):
    target = np.array(current, dtype=float, copy=True)
    if ('xyz' in value) == ('delta_xyz' in value):
        raise ValueError('Specify exactly one of xyz or delta_xyz')
    if 'delta_xyz' in value:
        if 'point_in_tcp' in value or 'approach' in value or 'jaw' in value:
            raise ValueError('Relative translation preserves orientation and cannot set point_in_tcp')
        delta = np.asarray(value['delta_xyz'], dtype=float)
        frame = value.get('frame', 'world')
        if delta.shape != (3,) or not np.isfinite(delta).all() or np.linalg.norm(delta) > .05 + 1e-12:
            raise ValueError('Relative translation must be finite and at most 5 cm')
        if frame not in ('world', 'tcp'):
            raise ValueError('Relative frame must be world or tcp')
        target[:3, 3] += target[:3,:3] @ delta if frame == 'tcp' else delta
    else:
        if 'frame' in value:
            raise ValueError('frame applies only to delta_xyz')
        target[:3, 3] = value['xyz']
    if ('approach' in value) != ('jaw' in value):
        raise ValueError('Specify both approach and jaw, or preserve current orientation by omitting both')
    if 'approach' in value:
        z, x = np.array(value['approach'], float), np.array(value['jaw'], float)
        if min(np.linalg.norm(z), np.linalg.norm(x)) < 1e-8:
            raise ValueError('Orientation axes must be nonzero')
        z /= np.linalg.norm(z)
        x /= np.linalg.norm(x)
        if abs(x @ z) > .05:
            raise ValueError('Approach and jaw directions must be perpendicular')
        x -= z*(x @ z)
        x /= np.linalg.norm(x)
        target[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    if 'point_in_tcp' in value:
        point = np.asarray(value['point_in_tcp'], dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('point_in_tcp must be a finite three-vector in TCP metres')
        target[:3, 3] -= target[:3, :3] @ point
    if not np.isfinite(target).all():
        raise ValueError('Pose must be finite')
    return target.tolist()


class RoboDojoHost:
    def __init__(self, args, config, binding, *, transport=None):
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.args, self.config, self.binding = args, config, binding
        self.experience_library = None
        experience_path = getattr(args, 'experience_bundle', None)
        experience_sha = getattr(args, 'experience_sha256', None)
        if bool(experience_path) != bool(experience_sha):
            raise ValueError('Experience requires both a manifest and its SHA256')
        if experience_path:
            from tools.harness_experience import ExperienceLibrary
            self.experience_library = ExperienceLibrary(experience_path, experience_sha,
                embodiment=config.value['components']['embodiment'])
            write_once(self.output/'experience-binding.json', self.experience_library.record())
        self.skill_library = SkillLibrary(config.value['components']['interface']['source_sha256'])
        self.transport = transport or StreamableHttpSimulatorMcpTransport(args.sim_url)
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.closed = self.stopping = False
        self.owned = {}
        self.cleanup = None
        self.tool_calls = 0
        self.packet = None
        self.error = None
        self.reason = None
        self._received_observation = False
        self._shown_images = {}
        roles = list(config.value['components']['embodiment']['effectors'])
        role = {'type': 'string', 'enum': roles}
        target = object_schema({'xyz': vector_schema(), 'approach': vector_schema(), 'jaw': vector_schema(),
                                'point_in_tcp':vector_schema()}, ['xyz'])
        definitions = {
            'read_skill': ('Read a fixed packaged workflow and its scoped contact contracts. Documentation only: no simulator request, physics advancement, file paths, hidden state or new capability. Choose by current interaction; costs one tool request, zero native actions.',
                object_schema({'skill_id': {'type':'string','enum':list(BUNDLES)}}, ['skill_id'])),
            'episode_status': ('Read the bound task, measured TCP axes, capabilities and official outcome.', object_schema()),
            'observe': ('Inspect current RGB images and measured robot state; no physics advancement.',
                        object_schema({'cameras': {'type':'array','items':{'type':'string'}}})),
            'mark_point': ('Select one original-image pixel. Returns the measured RGB-D surface point in world metres and a marked crop of the same public RGB frame for visual verification. Optional relative_to_tcp returns that point expressed in the selected measured TCP frame, using only this public point and robot pose; it does not identify an object or establish attachment. Preview coordinates are local to the crop; future pixel arguments still use original-image coordinates.',
                object_schema({'camera': {'type':'string'}, 'pixel': {'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2},
                               'relative_to_tcp':role}, ['camera','pixel'])),
            'move_to': ('Move the selected TCP through optional waypoints. xyz uses world metres. With optional target.point_in_tcp, xyz instead specifies the desired world location of that explicitly supplied TCP-frame point: the Host computes the TCP translation and reports it. This is a rigid-coordinate assumption, not measurement of a held object; remeasure if the grip changes. Each waypoint declares its own point_in_tcp or defaults to the TCP origin. approach is TCP +Z; jaw is +X. Omit both axes to preserve orientation. contact_mark permits only selected fingers contacting a freshly marked body. support_contact permits bounded placement and optional stationary support retention. push_contact permits bounded nonprehensile pushing using explicitly selected fingers, a fresh body_mark, surface_mark and target_marks. No grasp, closure or lift is required; objects may slip relative to the hand. At most 5cm translation in any direction with unchanged orientation, no waypoints/batch/other contact modes. Use for approach, pushing or withdrawal; it does not guarantee object motion or pulling adhesion. Current gripper command is preserved. Verify effect in images after every segment. tool_contact permits explicit monitored contact among a held tool, selected workpieces and the marked table patch. mode=stroke allows at most 5cm translation and 3mm height change with unchanged orientation. mode=adjust (reorient is an alias) allows at most 3cm TCP translation in any direction and 15 degrees rotation for contact height changes, angle changes, or both, within a TCP line segment and shortest orientation arc; it does not guarantee an object pivot or continued table support. Both modes keep contact, grasp, and collision checks; no waypoints or batch contact moves. Select workpieces with fresh target_marks or previously returned workpiece_references handles in target_handles (at most eight combined). Handles identify previously selected bodies, not current poses; they expire on release or grasp reauthorization and cannot cross roles or episodes. Empty marks and handles permit table-only sliding.',
                object_schema({'role':role,'target':target,'waypoints':{'type':'array','items':target,'maxItems':4},
                               'contact_mark':{'type':'string'},
                               'push_contact':object_schema({'body_mark':{'type':'string'},'surface_mark':{'type':'string'},
                                   'target_marks':{'type':'array','items':{'type':'string'},'maxItems':8},
                                   'fingers':{'type':'array','items':{'type':'string','enum':['finger_1','finger_2']},'minItems':1,'maxItems':2,'uniqueItems':True},
                                   'patch_radius_m':{'type':'number','minimum':.05,'maximum':.35}},['body_mark','surface_mark','target_marks','fingers']),
                               'support_contact':object_schema({'mode':{'type':'string','enum':['place']},
                                   'patch_radius_m':{'type':'number','minimum':.05,'maximum':.35},
                                   'surface_mark':{'type':'string'},'retain_support':{'type':'boolean'}},['mode','surface_mark']),
                               'tool_contact':object_schema({'mode':{'type':'string','enum':['stroke','adjust','reorient']},
                                   'patch_radius_m':{'type':'number','minimum':.05,'maximum':.35},
                                   'surface_mark':{'type':'string'},'target_marks':{'type':'array','items':{'type':'string'},
                                       'minItems':0,'maxItems':8},
                                   'target_handles':{'type':'array','items':{'type':'string','minLength':1},
                                       'minItems':0,'maxItems':8}},['mode','surface_mark','target_marks'])}, ['role','target'])),
            'gripper_control': ('Set measured parallel-gripper openness: 0 closed, 1 open. Inspect closure and lift evidence; closure alone does not confirm grasp.',
                object_schema({'role':role,'openness':{'type':'number','minimum':0,'maximum':1},
                               'contact_mark':{'type':'string'}}, ['role','openness'])),
            'finish_episode': ('Close the owned episode. success=true requires official benchmark success.',
                object_schema({'success':{'type':'boolean'},'reason':{'type':'string'}}, ['success','reason'])),
        }
        if config.value['components']['execution'].get('contact_policy') == 'physical_contact_v1':
            target['required'] = []
            target['properties'].update(delta_xyz=vector_schema(), frame={'type':'string','enum':['world','tcp']})
            target['oneOf'] = [{'required':['xyz']}, {'required':['delta_xyz']}]
            definitions['wait'] = (
                'Advance 0 < duration_s <= 2 seconds through billed native actions and per-physics-step checks. '
                'Hold arm joints measured at wait start and preserve gripper intent. Native action-period rounding is reported. '
                'Scene dynamics and task events continue; terminal/budget/physical checks may stop early. '
                'This is not force control or a task-stage switch.',
                object_schema({'duration_s':{'type':'number','exclusiveMinimum':0,'maximum':2}}, ['duration_s']))
            props = definitions['move_to'][1]['properties']
            props['position_tolerance_m'] = {'type':'number','minimum':.0005,'maximum':.003}
            for key in ('contact_mark','push_contact','support_contact','tool_contact'):
                props.pop(key, None)
            definitions['gripper_control'][1]['properties'].pop('contact_mark',None)
            definitions['move_to'] = (
                'Execute the requested TCP targets and up to four waypoints. xyz uses world metres; '
                'approach/jaw set TCP +Z/+X, omitted axes keep orientation. point_in_tcp specifies an explicit TCP-frame point. '
                'Alternatively delta_xyz plus frame=world or tcp gives a translation of at most 5 cm, preserving orientation. Each waypoint increment is relative to the preceding requested target, the first to measured TCP. '
                'No push/place/grasp intent or contact mark is required. Finger/environment contact is monitored with a geometric penetration bound; '
                'other robot collisions remain checked. Rotations, waypoints and dual-arm motions use the same generic interface. '
                'The Host does not search alternate spatial routes. Collision or IK failure returns evidence for you to replan. '
                'Optionally request position_tolerance_m in [0.0005,0.003]; default 0.003. This is a completion requirement, not a promise of accuracy. Unmet precision is reported without changing the target. '
                'No force control, grasp proof or held-object trajectory prediction is available. '
                'Inspect images after each action and use short segments for contact. Skills are optional guidance.',
                definitions['move_to'][1])
            definitions['gripper_control'] = (
                'Set gripper openness in [0,1]. No grasp/placement intent, mark or support proof is required. '
                'Report measured opening and any physical stop; closure does not prove grasp. Arm targets remain held.',
                definitions['gripper_control'][1])
        if config.value['components']['execution'].get('deformable_mode') == 'cloth_short_horizon_v1':
            for tool_name in ('move_to','gripper_control'):
                definitions[tool_name][1]['properties']['cloth_contact']={'type':'string','enum':['surface','local_layers']}
        if len(roles)>1:
            description,single = definitions['move_to']
            import copy
            motion_schema = copy.deepcopy(single)
            precision_schema = motion_schema['properties'].pop('position_tolerance_m', None)
            batch = object_schema({'motions':{'type':'array','minItems':1,'maxItems':len(roles),'items':motion_schema},
                                   'coordination':{'type':'string','enum':['independent','synchronized']}}, ['motions'])
            if precision_schema is not None:
                batch['properties']['position_tolerance_m'] = precision_schema
            definitions['move_to'] = (description+' Supply motions for both arms to allow automatic concurrent, delayed or sequential execution with adjustment feedback. Use coordination="synchronized" when independent timing would change the task intent. Omitted coordination defaults to independent for two-arm requests. Never supply start times.',
                                       {'type':'object','oneOf':[single,batch]})
        if config.value['components']['execution'].get('deformable_mode') == 'cloth_short_horizon_v1':
            desc,schema=definitions['move_to']
            definitions['move_to']=(desc+' CLOTH PROFILE: contact_mark defaults to a single connected material patch. Set cloth_contact="local_layers" with a fresh contact_mark when you intend to pinch local overlapping cloth layers: a 5cm radius, 12mm half-depth cylinder about the marked surface along the requested TCP approach, limited to 0.025m² selected surface area. Material identities freeze at authorization; new folds are not automatically added. Only the selected hand fingers are authorized. Host executes a short first segment (12cm/0.5rad without contact; 5cm/0.25rad with contact), reports effective targets and continuation_required if truncated; inspect images before continuing. Shared cloth keeps synchronized timing. Rigid support_contact/tool_contact are unavailable. Every physics step monitors measured cloth; future deformation is not predicted.',schema)
        if config.value['components']['execution'].get('deformable_mode') == 'cloth_visual_short_horizon_v2':
            desc,schema=definitions['move_to']
            definitions['move_to']=(desc+' VISUAL CLOTH PROFILE: fingers may interact with cloth without a contact mark or intent label. Only the first segment up to 5cm/0.25rad executes; effective targets and remaining work are reported. All multi-arm requests use synchronized timing, overriding independent with explicit feedback because shared material constraints are not measured. No automatic spatial detours. Inspect returned images after every segment. Private per-step cloth geometry is used for safety, never grasp detection; no grasp-loss sensor or force feedback is available.',schema)
        if config.value['components']['execution'].get('contact_policy') == 'physical_contact_v1':
            # Put the ceiling at request scope, including dual-arm batches; a
            # per-arm deadline would silently change coordination semantics.
            for name in ('move_to', 'gripper_control'):
                desc, schema = definitions[name]
                variants = schema.get('oneOf', [schema])
                for variant in variants:
                    variant['properties']['max_duration_s'] = {
                        'type':'number', 'exclusiveMinimum':0, 'maximum':2}
                definitions[name] = (desc+' Optional max_duration_s caps simulated execution time, rounded DOWN to whole native actions; values shorter than one action are rejected. It may return before the target or gripper command is achieved. Planning/network time is excluded. No automatic resume; inspect measured state. Gripper intent stays latched and can continue during later move/wait calls. In dual-arm batches the ceiling applies to the entire checked schedule, including any scheduling delay; one or both arms may be incomplete or unstarted when it returns. Omit the optional ceiling for normal longer repositioning; its supported range remains at most 2 seconds.', schema)
        if config.value['components']['benchmark'].get('completion_feedback'):
            definitions['check_task_completion'] = (
                'Request a read-only official-condition check when you think the task is complete or after a correction. '
                'This opt-in simulator oracle returns semantic deviations, not object coordinates, thresholds or motion plans. '
                'Costs one Host request, zero physical actions. Feedback is valid only at its reported action index. '
                'Pending or not-evaluated requirements are not confirmed failures. Generic feedback summarizes native history '
                'and the latest native evaluation, not a fresh evaluation of all scene geometry. '
                'A satisfied snapshot or recorded requirement does not award official success; use measured official outcome to finish. '
                'Do not assume a stable structure needs rebuilding before reviewing this feedback.', object_schema())
        if self.experience_library is not None:
            definitions['review_experience'] = (
                'Read a fixed historical visual/action experience. Omit segment_id to list the available segments; select an ID for before/after images, measured robot state, the recorded request and feedback. Images for each segment are delivered once. Read-only, no physics, one Host request. Optional reference, never a required motion prerequisite. Historical coordinates/marks/grasp evidence do not carry over; observe the live scene before planning.',
                object_schema({'segment_id':{'type':'integer','minimum':1,'maximum':12}}))
        self.completion_checks = 0
        self.schemas = {name: Tool(name=name, description=description, inputSchema=schema)
                        for name,(description,schema) in definitions.items()}
        try:
            created = self.transport.call_tool('create_env', {'env_id':args.env_id,'seed':args.seed,
                                                'include_objects':False}, timeout_s=240)
            if created.get('error'):
                raise RuntimeError('RoboDojo environment creation failed: '+str(created['error']))
            self.owned = {key:created[key] for key in ('session_id','handle')}
            write_once(self.output/'owned-environment.json',{
                **self.owned,'sim_url':args.sim_url,'env_id':args.env_id,
                'recovery':'After client interruption inspect/close this owned environment; never replay incomplete actions.'})
            reset = self.transport.call_tool('reset_env', {**self.owned,'seed':args.seed}, timeout_s=180)
            if reset.get('error'):
                raise RuntimeError('RoboDojo reset failed')
            self._remote({'operation':'status'})
            self._validate_binding()
            (self.output/'harness-binding.json').write_text(json.dumps({**binding,
                'resolved_config':config.value,'deployment':self.packet['control'],'status':'bound'},indent=2)+'\n')
            (self.output/'initial-observation.json').write_text(json.dumps(self.packet)+'\n')
            self._write_status()
        except BaseException:
            self.close('startup_failed')
            raise

    def _validate_binding(self):
        p = self.packet
        if p['control']['controller_id'] != 'robodojo.checked_joint_ik.v1':
            raise ValueError('Live controller differs from bound execution profile')
        expected_feedback=self.config.value['components']['benchmark'].get('completion_feedback')
        if expected_feedback != p['control'].get('completion_feedback'):
            raise ValueError('Live completion feedback differs from bound profile')
        budget_override=self.config.value['components']['benchmark'].get('episode_budget_override')
        if budget_override:
            from sim.envs.robodojo.episode_protocol import BUDGET_PROFILES
            expected = BUDGET_PROFILES[budget_override]
            budget=p['control'].get('episode_budget',{})
            protocol=budget.get('protocol') or {}
            if expected.get('unlimited'):
                limits = {item['source']: item for item in budget.get('limits', [])}
                if (protocol.get('override_id') != budget_override
                        or protocol.get('evaluation_variant') != 'unlimited_action_budget'
                        or protocol.get('comparable_to_native_budget_baseline') is not False
                        or budget.get('unlimited') is not True
                        or 'action_limit' not in budget or budget['action_limit'] is not None
                        or 'remaining_actions' not in budget or budget['remaining_actions'] is not None
                        or any(type(protocol.get(key)) is not int or protocol[key] <= 0 for key in
                               ('original_native_action_limit', 'original_adapter_action_limit'))
                        or any(key not in protocol or protocol[key] is not None for key in
                               ('effective_native_action_limit', 'effective_adapter_action_limit'))
                        or any(source not in limits or 'action_limit' not in limits[source]
                               or limits[source]['action_limit'] is not None
                               or 'remaining_actions' not in limits[source]
                               or limits[source]['remaining_actions'] is not None for source in
                               ('openeta_adapter_max_episode_steps', 'robodojo_native_step_lim'))):
                    raise ValueError('Live episode budget differs from unlimited-action profile')
            else:
                if (protocol.get('override_id') != budget_override
                        or protocol.get('original_native_action_limit') != expected['native_original']
                        or protocol.get('effective_native_action_limit') != expected['limit']
                        or protocol.get('comparable_to_native_budget_baseline') is not False):
                    raise ValueError('Live episode budget differs from extended-budget profile')
                if 'adapter_limit' in expected:
                    limits={item['source']:item.get('action_limit') for item in budget.get('limits',[])}
                    if (protocol.get('original_adapter_action_limit') != expected['adapter_original']
                            or protocol.get('effective_adapter_action_limit') != expected['adapter_limit']
                            or limits.get('openeta_adapter_max_episode_steps') != expected['adapter_limit']
                            or limits.get('robodojo_native_step_lim') != expected['limit']):
                        raise ValueError('Live adapter/native quotas differ from extended-budget profile')
        expected_init = self.config.value['components']['benchmark'].get('initialization_profile')
        initialization = p['control'].get('initialization') or {}
        if initialization.get('profile') != expected_init:
            raise ValueError('Live initialization differs from bound experiment')
        if expected_init and (initialization.get('completed_steps') != 300
                or initialization.get('native_stability_passed') is not True
                or initialization.get('phase') != 'passed'
                or initialization.get('comparable_to_native_initialization_baseline') is not False):
            raise ValueError('Initialization settling receipt is missing or incomplete')
        expected_runtime = self.config.value['components']['execution'].get('simulation_profile')
        runtime = p['control'].get('simulation_runtime') or {}
        if expected_runtime:
            if (runtime.get('profile') != expected_runtime or runtime.get('physics_tensor_device') != 'cuda:0'
                    or runtime.get('use_fabric') is not True):
                raise ValueError('Live GPU simulation runtime differs from bound profile')
        elif runtime.get('physics_tensor_device', 'cpu') != 'cpu':
            raise ValueError('GPU runtime cannot use an implicit CPU baseline profile')
        expected_physics=self.config.value['components']['execution'].get('physics_profile')
        physics=p['control'].get('physics_profile') or {}
        if physics.get('profile') != expected_physics:
            raise ValueError('Live cloth physics differs from bound experiment')
        if expected_physics and (physics.get('comparable_to_native_physics_baseline') is not False or not physics.get('materials')):
            raise ValueError('Cloth friction readback is missing')
        expected_contact=self.config.value['components']['execution'].get('contact_policy')
        if p['control']['collision_policy'].get('contact_policy') != expected_contact:
            raise ValueError('Live contact policy differs from bound execution profile')
        expected_mode=self.config.value['components']['execution'].get('deformable_mode')
        if expected_mode and p['control']['collision_policy'].get('deformables') != expected_mode:
            raise ValueError('Live cloth monitoring differs from bound execution contract')
        body = self.config.value['components']['embodiment']
        actual = p['control'].get('source_sha256',{})
        for name,expected in self.config.value['components']['interface']['source_sha256'].items():
            if name.startswith('sim/') and actual.get(name) != expected:
                raise ValueError('Live execution source differs from bound profile: '+name)
        if set(p['control']['roles']) != set(body['effectors']):
            raise ValueError('Live effector roles differ from configuration')
        layout = p['control'].get('layout_contract')
        if (not layout or layout.get('schema') != 'openeta.robodojo_layout.v1'
                or layout.get('execution_variant') != body['variant']
                or layout['effective']['layout'] != body.get('native_layout', body['variant'])
                or set(layout['effective']['target_roles']) != set(body['effectors'])
                or layout['effective']['support_actors'] != body.get('support_actors', {})
                or layout.get('support_policy') != self.config.value['components']['execution'].get('support_policy')
                or layout.get('missing_capabilities')):
            raise ValueError('Live native/effective robot layout is unverified or unsupported')
        observation = p['observation']
        normalize = lambda text:re.sub(r'[^a-z0-9]+',' ',text.lower()).strip()
        if normalize(observation.get('task','')) != normalize(self.args.task):
            raise ValueError('Requested task text differs from the official seeded task instruction')
        if observation.get('objects'):
            raise ValueError('Privileged scene annotations must not enter this Host')
        cameras = camera_directory(observation)
        if not cameras or not all(c['pixel_to_world_available'] for c in cameras.values()):
            raise ValueError('Live RGB-D calibration is incomplete')
        root = observation['robot']
        entries = root.get('arms', {'primary':root})
        for role, definition in body['effectors'].items():
            entry = entries[role]
            namespace = body['groups'][definition['group']].get('articulation')
            if namespace is not None and p['control'].get('articulations',{}).get(role,{}).get('id') != namespace:
                raise ValueError('Live articulation identity differs from joint resource namespace')
            if entry.get('joint_names') != body['groups'][definition['group']]['joint_names']:
                raise ValueError('Live joint identity differs from configuration')
            if entry.get('metadata', {}).get('robot_contract') != 'openeta.x5_parallel_link6_tcp.v1':
                raise ValueError('Live TCP contract is unverified')
            from sim.envs.robodojo.scene_geometry import pose_matrix
            from sim.envs.robodojo.x5_calibration import TCP_FROM_LINK
            pose = entry['end_effector_pose']
            q = pose['quat_xyzw']
            reported = pose_matrix([*pose['xyz'],q[3],*q[:3]])
            if not np.allclose(reported @ TCP_FROM_LINK,p['control']['tcp_poses'][role],atol=2e-6):
                raise ValueError('Reported pose and executed TCP frame disagree')
        official_outcome(p)

    def _remote(self, request):
        # Worker motion is bounded at 360s. Leave room for the last checked
        # native step, observation encoding and the final stop receipt.
        if self.config.value['components']['execution'].get('deformable_mode'):
            request={**request,'deformable_mode':self.config.value['components']['execution']['deformable_mode']}
        if self.config.value['components']['execution'].get('contact_policy'):
            request={**request,'contact_policy':self.config.value['components']['execution']['contact_policy']}
        physics_profile=self.config.value['components']['execution'].get('physics_profile')
        if physics_profile:
            request={**request,'physics_profile':physics_profile}
        simulation_profile=self.config.value['components']['execution'].get('simulation_profile')
        if simulation_profile:
            request={**request,'simulation_profile':simulation_profile}
        initialization_profile=self.config.value['components']['benchmark'].get('initialization_profile')
        if initialization_profile:
            request={**request,'initialization_profile':initialization_profile}
        budget_override=self.config.value['components']['benchmark'].get('episode_budget_override')
        if budget_override:
            request={**request,'episode_budget_override':budget_override}
        completion_profile=self.config.value['components']['benchmark'].get('completion_feedback')
        if completion_profile:
            request={**request,'completion_feedback':completion_profile}
        timeout_s = 420 if request.get('operation') == 'move' else 240
        response = self.transport.call_tool('configured_robodojo', {**self.owned,'request':request}, timeout_s=timeout_s)
        if response.get('error'):
            if response.get('state_valid') is False:
                self.stopping = True
                raise RemoteExecutionUnknown(str(response))
            raise ValueError(str(response))
        if response.get('observation'):
            self.packet = response
            self._received_observation = True
            (self.output/'last-observation.json').write_text(json.dumps(response)+'\n')
        return response

    def _status(self):
        outcome = official_outcome(self.packet) if self.packet else {'status':'unknown','benchmark_success':None}
        return {'config_id':self.config.id, 'task':self.args.task, 'closed':self.closed,
                'tool_calls':self.tool_calls,'official_task_success':outcome['benchmark_success'] is True,
                'outcome':outcome,'cleanup':self.cleanup,'error':self.error,'reason':self.reason,
                'simulation_budget':self.packet['control'].get('episode_budget') if self.packet else None,
                'physics_profile':self.packet['control'].get('physics_profile') if self.packet else None,
                'initialization':self.packet['control'].get('initialization') if self.packet else None,
                'simulation_runtime':self.packet['control'].get('simulation_runtime') if self.packet else None,
                'completion_feedback':self.config.value['components']['benchmark'].get('completion_feedback'),
                'completion_checks':self.completion_checks,
                'elapsed_s':time.monotonic()-self.started,
                'remaining_requests':max(0,self.args.max_requests-self.tool_calls),
                'limits':{'max_requests':self.args.max_requests,'episode_timeout_s':self.args.timeout or None}}

    def _write_status(self):
        (self.output/'host-status.json').write_text(json.dumps(self._status(),indent=2)+'\n')

    def _frame_signature(self, frame):
        fields=('rgb_base64','depth_base64','width','height','intrinsics','extrinsics')
        digest=hashlib.sha256(json.dumps({k:frame.get(k) for k in fields},sort_keys=True).encode()).hexdigest()
        return self.packet['action_index'],digest

    def _images_match_last_shown(self):
        frames=self.packet['observation']['cameras'] if self.packet else []
        return bool(frames) and all(frame.get('rgb_base64') and
            self._shown_images.get(frame['frame_id'])==self._frame_signature(frame) for frame in frames)

    def _mark_preview(self, mark, journal):
        """Visual confirmation from public RGB only; no semantic annotations."""
        import base64
        import io
        import os
        from PIL import Image, ImageDraw
        try:
            frame = next(f for f in self.packet['observation']['cameras']
                         if f['frame_id'] == mark['camera'])
            if mark['source_action_index'] != self.packet['action_index']:
                raise ValueError('Mark does not match the current public frame')
            raw = base64.b64decode(frame['rgb_base64'], validate=True)
            original = Image.open(io.BytesIO(raw)).convert('RGB')
            width, height = original.size
            x, y = mark['pixel']
            if (original.size != (frame['width'], frame['height']) or
                    not 0 <= x < width or not 0 <= y < height):
                raise ValueError('Invalid public image dimensions or pixel')
            crop_width, crop_height = min(160, width), min(160, height)
            left = max(0, min(x-crop_width//2, width-crop_width))
            top = max(0, min(y-crop_height//2, height-crop_height))
            box = (left, top, left+crop_width, top+crop_height)
            crop = original.crop(box)
            px, py = x-left, y-top
            draw = ImageDraw.Draw(crop)
            # Keep the selected pixel and its immediate neighbourhood visible.
            lines = [(px-13, py, px-5, py), (px+5, py, px+13, py),
                     (px, py-13, px, py-5), (px, py+5, px, py+13)]
            for line in lines:
                draw.line(line, fill=(0, 0, 0), width=3)
                draw.line(line, fill=(255, 255, 0), width=1)
            buffer = io.BytesIO()
            crop.save(buffer, format='PNG')
            data = buffer.getvalue()
        except (KeyError, StopIteration, ValueError, OSError, TypeError):
            return {'status': 'unavailable', 'reason': 'public_rgb_preview_unavailable',
                    'mark_unchanged': True}, []
        # An evidence write failure is handled as such by the caller; do not
        # silently deliver an unarchived derived image.
        with (journal.path/'mark-preview.png').open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {'status': 'available', 'camera': mark['camera'],
            'source_action_index': mark['source_action_index'], 'original_pixel': [x, y],
            'crop_box_xyxy': list(box), 'selected_pixel_in_crop': [px, py],
            'source_rgb_sha256': hashlib.sha256(raw).hexdigest(),
            'artifact': {'path': 'mark-preview.png', 'sha256': hashlib.sha256(data).hexdigest()},
            'derivation': 'public_rgb_crop_with_gapped_crosshair; no_rescale',
            'semantic_identification': False}
        return metadata, [TextContent(type='text', text=(
            f'Mark preview for {mark["camera"]}, original pixel ({x}, {y}), '
            f'action_index={mark["source_action_index"]}. Crop origin=({left}, {top}); '
            f'selected crop pixel=({px}, {py}) is between the yellow crosshair arms. '
            'This is the same public RGB image, not object identification. '
            'Use original-image coordinates for subsequent marks.')),
            ImageContent(type='image', data=base64.b64encode(data).decode(), mimeType='image/png')]

    def _result(self, *, receipt=None, error=None, images=True, cameras=None, evidence=None, image_delivery=None):
        status = self._status()
        result = {'status':status}
        feedback = motion_feedback(receipt)
        if feedback is not None:
            result['motion_feedback'] = feedback
        result.update(receipt=receipt,error=error)
        if image_delivery is not None:
            result['image_delivery']=image_delivery
        if evidence is not None:
            result['evidence'] = evidence
        frames = []
        if self.packet:
            result['action_index'] = self.packet['action_index']
            result['tcp_state'] = {role: {'xyz':np.array(t)[:3,3].tolist(),
                                          'approach':np.array(t)[:3,2].tolist(),
                                          'jaw':np.array(t)[:3,0].tolist()}
                                   for role,t in self.packet['control']['tcp_poses'].items()}
            result['robot'] = self.packet['observation']['robot']
            result['collision_policy'] = self.packet['control']['collision_policy']
            result['observation_clock'] = self.packet['control'].get('observation_clock', {'status':'unavailable'})
            result['execution_capabilities'] = {
                'bounded_wait': {'available':'wait' in self.schemas,'max_duration_s':2.,
                                 'billed':True,'hold':'measured_arm_configuration_and_latched_gripper_intent'},
                'relative_translation': {'available':'wait' in self.schemas,'max_norm_m':.05,
                                         'frames':['world','tcp'],'precision_improved':False},
                'motion_completion_tolerance':{'position_m':.003,'orientation_rad':.03,
                                               'position_request_range_m':[.0005,.003] if 'wait' in self.schemas else None,
                                               'insertion_or_contact_success_inferred':False},
                'force_feedback':{'available':False,'reason':'sensor_load_chain_not_quantitatively_calibrated'},
                'force_control':{'available':False},
                'dynamic_world_prediction':{'available':False,'monitor':'measured_world_poses_each_physics_step'},
                'fluid_collision':{'available':False},
                'automatic_spatial_replanning':{'available':False}}
            result['active_support_holds'] = self.packet['control'].get('active_support_holds',[])
            result['layout_contract'] = self.packet['control'].get('layout_contract')
            if self.packet['control'].get('physics_profile'):
                result['physics_profile'] = self.packet['control']['physics_profile']
            if self.packet['control'].get('simulation_runtime'):
                result['simulation_runtime'] = self.packet['control']['simulation_runtime']
            result['cameras'] = []
            for frame in self.packet['observation']['cameras']:
                name = frame['frame_id']
                result['cameras'].append({'name':name,'width':frame['width'],'height':frame['height']})
                if images and (cameras is None or name in cameras) and frame.get('rgb_base64'):
                    freshness = (evidence or {}).get('observation', 'cached')
                    frames += [TextContent(type='text',text=f'{name}: original {frame["width"]}×{frame["height"]} image, action_index={self.packet["action_index"]}, observation={freshness}'),
                               ImageContent(type='image',data=frame['rgb_base64'],mimeType='image/png')]
                    if self.config.value['components']['interface'].get('visual_aids') == 'tcp_overlay_v1':
                        from tools.robodojo_visual_aids import render_tcp_overlay
                        annotated, aid = render_tcp_overlay(frame, self.packet['control']['tcp_poses'])
                        result.setdefault('visual_aids', []).append(aid)
                        if annotated:
                            frames += [TextContent(type='text', text=f'{name}: derived TCP overlay, same action_index and freshness as preceding original. ' + aid['notice']),
                                       ImageContent(type='image', data=annotated, mimeType='image/png')]
        return [TextContent(type='text',text=json.dumps(result,ensure_ascii=False)), *frames]

    def call(self, name, arguments):
        with self.lock:
            self.tool_calls += 1
            self._received_observation = False
            receipt = error = None
            target_conversions = []
            historical_blocks = []
            try:
                journal = CallEvidence(self.output, self.tool_calls, name, arguments,
                                       self.config.sha256, self.packet)
            except Exception as exc:
                self.stopping = True
                return self._result(error={'code':'evidence_write_failed_before_dispatch',
                                           'type':type(exc).__name__, 'action_dispatched':False},
                                    images=False)
            try:
                if name not in self.schemas:
                    raise ValueError('Unknown configured tool')
                jsonschema.validate(arguments, self.schemas[name].inputSchema)
                if self.closed:
                    raise ValueError('Episode is closed')
                if (self.stopping or (self.args.timeout > 0 and time.monotonic()-self.started > self.args.timeout)
                        or self.tool_calls > self.args.max_requests):
                    self.close('episode_budget_or_stop')
                    raise ValueError('Episode budget exhausted or stop requested')
                if name == 'finish_episode':
                    if arguments['success'] and not self._status()['official_task_success']:
                        raise ValueError('Official success has not been established')
                    self.close(arguments['reason'])
                elif name == 'review_experience':
                    receipt, historical_blocks = self.experience_library.read(arguments.get('segment_id'))
                elif name == 'read_skill':
                    receipt = self.skill_library.read(arguments['skill_id'], physical_contact=
                        self.config.value['components']['execution'].get('contact_policy') == 'physical_contact_v1',
                        visual_cloth=self.config.value['components']['execution'].get('deformable_mode') == 'cloth_visual_short_horizon_v2')
                elif name == 'check_task_completion':
                    self.completion_checks += 1
                    response=self._remote({'operation':'check_completion','action_index':self.packet['action_index']})
                    receipt=response.get('receipt')
                elif name in ('episode_status','observe'):
                    self._remote({'operation':'status'})
                elif name == 'wait':
                    response = self._remote({'operation':'wait', 'duration_s':arguments['duration_s'],
                                             'action_index':self.packet['action_index']})
                    receipt = response.get('receipt')
                    if response.get('admitted') is False:
                        error = {'code':'motion_not_admitted','detail':receipt}
                elif name == 'mark_point':
                    response = self._remote({'operation':'mark','action_index':self.packet['action_index'],
                                             'camera':arguments['camera'],'pixel':arguments['pixel']})
                    receipt = dict(response['receipt'])
                    if 'relative_to_tcp' in arguments:
                        role = arguments['relative_to_tcp']
                        mark = receipt['mark']
                        if mark['source_action_index'] != self.packet['action_index']:
                            raise ValueError('Point and TCP must describe the same measured state')
                        tcp = np.asarray(self.packet['control']['tcp_poses'][role], dtype=float)
                        point = tcp[:3,:3].T @ (np.asarray(mark['xyz'], dtype=float)-tcp[:3,3])
                        if point.shape != (3,) or not np.isfinite(point).all():
                            raise ValueError('Point-frame conversion requires finite public coordinates')
                        receipt['tcp_point'] = {'role':role,'point_in_tcp':point.tolist(),
                            'source_action_index':mark['source_action_index'],
                            'source':'public_rgbd_mark_and_measured_robot_tcp_only',
                            'object_identity_or_attachment_inferred':False,
                            'validity':'coordinates_at_measurement; later_rigid_motion_is_an_agent_assumption'}
                else:
                    motions = arguments.get('motions') if name=='move_to' else None
                    if motions is not None and len({m['role'] for m in motions})!=len(motions):
                        raise ValueError('Each effector may appear only once in synchronized motion')
                    request = {'operation':'move' if name=='move_to' else 'gripper',
                               'action_index':self.packet['action_index']}
                    if 'max_duration_s' in arguments:
                        request['max_duration_s'] = arguments['max_duration_s']
                    if motions is None:
                        request['role'] = arguments['role']
                    selections=motions if motions is not None else [arguments]
                    for selection in selections:
                        if 'cloth_contact' in selection:
                            if not selection.get('contact_mark'):
                                raise ValueError('cloth_contact requires a fresh contact_mark')
                            request.setdefault('cloth_contacts',{})[selection['role']]=selection['cloth_contact']
                    if 'contact_mark' in arguments:
                        request['contact_mark'] = arguments['contact_mark']
                    if name == 'move_to':
                        if 'position_tolerance_m' in arguments:
                            request['position_tolerance_m'] = arguments['position_tolerance_m']
                        if motions is not None and any(any(k in m for k in ('support_contact','tool_contact','push_contact')) for m in motions):
                            raise ValueError('Support placement or tool stroke requires a separate single-arm move')
                        if 'push_contact' in arguments:
                            if any(k in arguments for k in ('support_contact','tool_contact','contact_mark','cloth_contact','waypoints')):
                                raise ValueError('Push contact cannot be combined with grasp/placement/tool contact or waypoints')
                            request['push_contact']=arguments['push_contact']
                        if 'support_contact' in arguments and 'tool_contact' in arguments:
                            raise ValueError('Choose placement or tool stroke, not both in one move')
                        if 'support_contact' in arguments:
                            request['support_contact']=arguments['support_contact']
                        if 'tool_contact' in arguments:
                            contact = arguments['tool_contact']
                            if len(contact['target_marks']) + len(contact.get('target_handles', [])) > 8:
                                raise ValueError('Tool contact accepts at most eight marks and workpiece references in total')
                            request['tool_contact']=arguments['tool_contact']
                        request['coordination'] = arguments.get('coordination',
                            'independent' if motions is not None and len(motions)>1 else 'synchronized')
                        request['targets'] = {}
                        for motion in motions if motions is not None else [arguments]:
                            current = self.packet['control']['tcp_poses'][motion['role']]
                            targets = []
                            for index,target in enumerate([*motion.get('waypoints',[]),motion['target']]):
                                reference = np.asarray(current, dtype=float)
                                current = target_matrix(target,current)
                                if 'delta_xyz' in target:
                                    target_conversions.append({'kind':'explicit_relative_translation',
                                        'role':motion['role'],'waypoint_index':index,
                                        'delta_xyz':target['delta_xyz'],'frame':target.get('frame','world'),
                                        'reference_tcp_pose':reference.tolist(),
                                        'reference_source':'measured_tcp' if index == 0 else 'preceding_requested_target',
                                        'effective_tcp_pose':current,'automatic_spatial_replanning':False})
                                targets.append(current)
                                if 'point_in_tcp' in target:
                                    target_conversions.append({'kind':'explicit_tcp_point_target_conversion',
                                        'role':motion['role'],'waypoint_index':index,
                                        'requested_world_point_xyz':target['xyz'],
                                        'point_in_tcp':target['point_in_tcp'],
                                        'effective_tcp_xyz':np.asarray(current)[:3,3].tolist(),
                                        'assumption':'supplied_point_is_fixed_in_the_commanded_tcp_frame',
                                        'actual_object_point_measured_after_motion':False,
                                        'scope':'coordinate_conversion_only; no_contact_authorization_or_success_claim'})
                            request['targets'][motion['role']] = targets
                            if motions is not None and motion.get('contact_mark'):
                                request.setdefault('contact_marks',{})[motion['role']] = motion['contact_mark']
                    else:
                        request['openness'] = arguments['openness']
                    response = self._remote(request)
                    receipt = response.get('receipt')
                    if response.get('admitted') is False:
                        error = {'code':'motion_not_admitted','detail':receipt}
            except (ValueError, KeyError, jsonschema.ValidationError) as exc:
                error = {'code':'request_rejected','message':str(exc)}
            except Exception as exc:
                error = {'code':'transport_or_execution_outcome_unknown','type':type(exc).__name__}
                if isinstance(exc, RemoteExecutionUnknown):
                    error['message'] = str(exc)
                self.error = error
                self.close('unknown_action_outcome')
            if target_conversions:
                receipt = dict(receipt or {})
                receipt['host_adjustments'] = [*receipt.get('host_adjustments',[]), *target_conversions]
            unknown = error and error['code']=='transport_or_execution_outcome_unknown'
            provenance = {'call_id':journal.call_id, 'path':f'calls/{journal.call_id}',
                          'observation':('outcome_unknown_last_known' if unknown else
                                         'received_this_call' if self._received_observation else 'cached'),
                          'after_observation_received':self._received_observation}
            omit_unchanged=(name=='move_to' and not unknown and receipt is not None
                and receipt.get('native_actions')==0 and receipt.get('physics_steps')==0
                and self._images_match_last_shown())
            include_images=name not in ('mark_point','read_skill','review_experience') and not omit_unchanged
            image_delivery=({'status':'omitted_unchanged','reference_action_index':self.packet['action_index'],
                'reason':'motion_not_executed_and_rgbd_calibration_match_last_shown','available_via':'observe'}
                if omit_unchanged else None)
            preview = []
            if name == 'mark_point' and error is None and (receipt or {}).get('mark'):
                try:
                    metadata, preview = self._mark_preview(receipt['mark'], journal)
                    receipt = {**receipt, 'mark_preview': metadata}
                except OSError as exc:
                    self.stopping = True
                    return self._result(receipt=receipt, error={'code': 'evidence_write_failed_after_dispatch',
                        'type': type(exc).__name__, 'automatic_retry_allowed': False}, images=False,
                        evidence=provenance)
            result = self._result(receipt=receipt,error=error,images=include_images,
                                  cameras=arguments.get('cameras') if isinstance(arguments,dict) else None,
                                  evidence=provenance,image_delivery=image_delivery)
            # A read-only mark need not repeat unchanged, already delivered
            # robot/configuration state. The full packet is still journalled.
            value = json.loads(result[0].text)
            repeated_fields = ('robot', 'collision_policy', 'layout_contract')
            state_signature = hashlib.sha256(json.dumps(
                {key:value.get(key) for key in repeated_fields}, sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()
            compact_mark = (name == 'mark_point' and error is None
                and state_signature == getattr(self, '_last_full_state_signature', None))
            if compact_mark:
                for key in repeated_fields:
                    value.pop(key, None)
                value['state_delivery'] = {'status':'omitted_unchanged_fields',
                    'fields':list(repeated_fields),
                    'reference':self._last_full_state_reference,
                    'available_via':['observe','episode_status']}
                result[0] = TextContent(type='text', text=json.dumps(value, ensure_ascii=False))
            result.extend(preview)
            result.extend(historical_blocks)
            if self.config.value['components']['interface'].get('image_encoding') == 'lossless_webp_if_smaller_v1':
                from tools.robodojo_image_transport import encode_content
                result, encoding = encode_content(result)
                value = json.loads(result[0].text)
                value['image_transport'] = encoding
                result[0] = TextContent(type='text', text=json.dumps(value, ensure_ascii=False))
            try:
                journal.complete(result, self.packet, received_observation=self._received_observation)
                self._write_status()
                with (self.output/'requests.jsonl').open('a') as stream:
                    stream.write(json.dumps({'index':self.tool_calls,'tool':name,'arguments':arguments,
                                             'receipt':receipt,'error':error,'status':self._status(),
                                             'evidence':provenance})+'\n')
            except Exception as exc:
                self.stopping = True
                return self._result(receipt=receipt, error={'code':'evidence_write_failed_after_dispatch',
                    'type':type(exc).__name__, 'previous_error':error, 'automatic_retry_allowed':False},
                    images=False, evidence=provenance)
            if include_images and self.packet:
                selected=arguments.get('cameras') if isinstance(arguments,dict) else None
                for frame in self.packet['observation']['cameras']:
                    if frame.get('rgb_base64') and (selected is None or frame['frame_id'] in selected):
                        self._shown_images[frame['frame_id']]=self._frame_signature(frame)
            if not compact_mark:
                self._last_full_state_signature = state_signature
                self._last_full_state_reference = {'call_id':journal.call_id,
                    'action_index':value.get('action_index')}
            return result

    def request_stop(self):
        self.stopping = True

    def close(self, reason='mcp_closed'):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.reason = reason
            if self.owned:
                try:
                    result = self.transport.call_tool('close_env',self.owned,timeout_s=90)
                    self.cleanup = {'ok':not result.get('error') and result.get('ok',True) is not False,'receipt':result}
                except Exception as exc:
                    self.cleanup = {'ok':False,'error_type':type(exc).__name__}
            self._write_status()
