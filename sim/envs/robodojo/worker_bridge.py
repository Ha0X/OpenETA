"""Private worker endpoint for a configuration-bound RoboDojo Host."""
import copy
import hashlib
import secrets
from pathlib import Path

import numpy as np

from .calibration import SENSOR_CONTRACT
from .outcome import snapshot


def episode_budget(adapter):
    protocol = getattr(adapter, '_episode_protocol', None) or {}
    if protocol.get('evaluation_variant') == 'unlimited_action_budget':
        from .episode_protocol import bind_protocol
        # Verify actual worker state, never infer no cap from a missing value.
        bind_protocol(adapter, protocol['override_id'], 'status')
        limits = [
            {'source': 'openeta_adapter_max_episode_steps', 'action_limit': None,
             'used_actions': int(adapter._step_count), 'remaining_actions': None},
            {'source': 'robodojo_native_step_lim', 'action_limit': None,
             'used_actions': int(adapter._env.take_action_cnt[0]), 'remaining_actions': None},
        ]
        return {**limits[0], 'limits': limits, 'native_limit_available': True,
                'unlimited': True, 'protocol': copy.deepcopy(protocol),
                'terminal': bool(adapter._terminal), 'unit': 'native_adapter_step_calls',
                'note': 'No episode native-action cap. Used actions are recorded; null limits mean unlimited. '
                        'Tool-call quota, per-call execution guards and native task terminal rules still apply.'}
    limit=int(adapter._max_episode_steps)
    used=int(adapter._step_count)
    limits=[{'source':'openeta_adapter_max_episode_steps','action_limit':limit,
             'used_actions':used,'remaining_actions':max(0,limit-used)}]
    # EvalEnv enforces its own task-specific limit, using its own counter.
    # Read it without changing either quota or exposing reward/scene state.
    native=getattr(adapter,'_env',None)
    native_limit=getattr(native,'step_lim',None)
    native_counts=getattr(native,'take_action_cnt',None)
    native_available=False
    if (isinstance(native_limit,(int,np.integer)) and not isinstance(native_limit,bool)
            and native_limit>0 and isinstance(native_counts,(list,tuple,np.ndarray))
            and np.shape(native_counts)==(1,)):
        native_used=native_counts[0]
        if (isinstance(native_used,(int,np.integer)) and not isinstance(native_used,bool)
                and native_used>=0):
            native_available=True
            limits.append({'source':'robodojo_native_step_lim','action_limit':int(native_limit),
                           'used_actions':int(native_used),
                           'remaining_actions':max(0,int(native_limit)-int(native_used))})
    effective=min(limits,key=lambda item:item['remaining_actions'])
    return {**effective,'limits':limits,'native_limit_available':native_available,
            'protocol':copy.deepcopy(getattr(adapter, '_episode_protocol', None)),
            'terminal':bool(adapter._terminal),
            'unit':'native_adapter_step_calls',
            'note':('Remaining actions are the minimum of the reported independent quotas. '
                    'Task success/failure may end the episode earlier; this is not a success prediction.'
                    if native_available else
                    'Native quota unavailable; only the adapter cap is reported. The benchmark may end earlier.')}


def operate(adapter, body):
    from adapter.protocol import EnvObservation
    from .checked_control import CheckedController
    if getattr(adapter._env, '_openeta_sensor_contract', None) != SENSOR_CONTRACT:
        raise ValueError('Calibrated RoboDojo deployment is required')
    if not adapter._state_valid:
        raise ValueError('Unknown action outcome; reset before continuing')
    from .cloth_material_profile import inspect_binding as inspect_material_binding
    material_profile = inspect_material_binding(adapter._env, body.get('physics_profile'))
    from .initialization_profile import inspect_binding as inspect_initialization_binding
    initialization = inspect_initialization_binding(adapter, body.get('initialization_profile'))
    from .episode_protocol import bind_protocol
    bind_protocol(adapter, body.get('episode_budget_override'), body.get('operation'))
    mode=body.get('deformable_mode')
    if mode not in (None,'cloth_short_horizon_v1','cloth_visual_short_horizon_v2'):
        raise ValueError('Unknown deformable execution contract')
    if getattr(adapter, '_checked_controller', None) is None:
        adapter._checked_controller = (CheckedController(adapter,allow_cloth=True,cloth_mode=mode) if mode else
            CheckedController(adapter,allow_articulations=True,allow_fluid=True) if body.get('contact_policy') == 'physical_contact_v1' else CheckedController(adapter))
        if hasattr(adapter, '_checked_contact_policy'):
            del adapter._checked_contact_policy
        adapter._checked_deformable_mode=mode
        adapter._checked_marks = {}
        adapter._checked_controller.source_sha256 = {
            'sim/envs/robodojo/'+p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob('*.py') if p.name!='__init__.py'}
    if mode != getattr(adapter,'_checked_deformable_mode',None):
        raise ValueError('Cannot change deformable execution contract within an episode')
    control = adapter._checked_controller
    policy = body.get('contact_policy')
    if policy not in (None, 'physical_contact_v1') or (policy and mode and mode!='cloth_visual_short_horizon_v2') or (mode=='cloth_visual_short_horizon_v2' and not policy):
        raise ValueError('Unsupported contact policy or deformable combination')
    if not hasattr(adapter, '_checked_contact_policy'):
        if policy and (body.get('operation') != 'status' or adapter._step_count != 0):
            raise ValueError('Contact policy must bind at initial status')
        adapter._checked_contact_policy = policy
        if policy:
            from .physical_contact import PhysicalContact
            control.physical_contact = PhysicalContact(control.world, roles=[r['role'] for r in control.world.scene.snapshot['robots']])
            control._rebuild_allowed_pairs()
    if policy != adapter._checked_contact_policy:
        raise ValueError('Cannot change contact policy within an episode')
    cloth=getattr(control,'cloth',None)
    op = body.get('operation')
    receipt = None
    completion_profile=body.get('completion_feedback')
    from . import completion_feedback as tower_feedback
    from . import completion_feedback_generic as generic_feedback
    if completion_profile is not None:
        if completion_profile == tower_feedback.PROFILE:
            if adapter._task_name != 'build_tower' or adapter._robot_variant != 'dual_x5':
                raise ValueError('Unsupported completion feedback profile')
        elif completion_profile == generic_feedback.PROFILE:
            if adapter._task_name not in generic_feedback.TASK_PINS:
                raise ValueError('Unsupported completion feedback task')
        else:
            raise ValueError('Unsupported completion feedback profile')
    if not hasattr(adapter, '_completion_profile'):
        if completion_profile and (op != 'status' or adapter._step_count != 0):
            raise ValueError('Completion profile must bind at initial status')
        adapter._completion_profile = completion_profile
        adapter._completion_trace = (generic_feedback.NativeTrace()
            if completion_profile == generic_feedback.PROFILE else None)
    if completion_profile != adapter._completion_profile:
        raise ValueError('Cannot change completion feedback within an episode')
    if op == 'check_completion':
        if completion_profile is None:
            raise ValueError('Completion feedback is not enabled')
        checker = generic_feedback if completion_profile == generic_feedback.PROFILE else tower_feedback
        receipt=checker.check(adapter,body.get('action_index'))
    elif op == 'mark':
        if body.get('action_index') != adapter._step_count:
            raise ValueError('Stale image: observe again before marking')
        camera = adapter._last_obs['cameras'][body['camera']]
        u, v = body['pixel']
        if type(u) is not int or type(v) is not int:
            raise ValueError('Use integer original-image pixels')
        depth = np.asarray(camera['depth'])
        if not 0 <= u < depth.shape[1] or not 0 <= v < depth.shape[0]:
            raise ValueError('Pixel is outside original image')
        z = float(depth[v, u])
        if not np.isfinite(z) or not .02 < z < 3:
            raise ValueError('Selected pixel has no valid optical depth')
        k, ext = camera['intrinsics'], camera['extrinsics']
        point = np.array(ext['camera_to_world']) @ [(u-k['cx'])*z/k['fx'], (v-k['cy'])*z/k['fy'], z, 1]
        token = 'mark_' + secrets.token_hex(6)
        mark = {'mark_id': token, 'camera': body['camera'], 'pixel': [u,v], 'xyz': point[:3].tolist(),
                'source_action_index': adapter._step_count, 'depth_m': z}
        adapter._checked_marks[token] = mark
        receipt = {'mark': mark}
    elif op in ('move', 'gripper', 'wait'):
        control.support_events = []
        control.motion_rate_events = {}
        control.arm_continuity_events = {}
        if hasattr(control, 'gripper_continuity'):
            control.gripper_continuity.events.clear()
        if adapter._terminal:
            raise ValueError('Official episode has ended')
        if body.get('action_index') != adapter._step_count:
            raise ValueError('Stale motion state: observe before issuing another command')
        if cloth is not None:
            cloth.recovery_events=[]
            cloth.pending_recovery=None
        physical = getattr(control, 'physical_contact', None)
        from .execution_horizon import prepare as prepare_horizon
        horizon = prepare_horizon(control, body)
        if physical is not None:
            if any(body.get(k) for k in ('contact_marks','contact_mark','cloth_contacts','push_contact','support_contact','tool_contact')):
                raise ValueError('This profile accepts generic motion/gripper commands; legacy contact authorization is not required or supported')
            control.world.refresh()
            physical.begin(control.world.positions())
        authorizations = []
        marks = dict(body.get('contact_marks',{}))
        if body.get('contact_mark'):
            marks[body['role']] = body['contact_mark']
        contact_modes=body.get('cloth_contacts',{})
        if not isinstance(contact_modes,dict) or set(contact_modes)-set(marks):
            raise ValueError('Cloth contact mode requires a fresh contact mark for that role')
        if contact_modes and cloth is None:
            raise ValueError('Cloth contact mode requires a cloth execution profile')
        if any(v not in ('surface','local_layers') for v in contact_modes.values()):
            raise ValueError('Unknown cloth contact mode')
        validated = []
        for role,token in marks.items():
            mark = adapter._checked_marks.get(token)
            if mark is None or mark['source_action_index'] != adapter._step_count:
                raise ValueError('Contact authorization requires a fresh marked surface')
            validated.append((role,mark['xyz']))
        previous_grants,previous_pairs = dict(control.contact_grants),list(control.allowed_pairs)
        previous_holds = dict(control.support_holds)
        try:
            for role,point in validated:
                if role in contact_modes:
                    pose=(body['targets'][role][0] if op=='move' else control.tcp(control.roles.index(role)))
                    pose=np.asarray(pose,dtype=float)
                    if pose.shape!=(4,4) or not np.isfinite(pose).all():
                        raise ValueError('Invalid contact approach pose')
                    authorization=control.authorize_contact(role,point,cloth_contact=contact_modes[role],approach=pose[:3,2])
                else:
                    authorization = control.authorize_contact(role,point)
                authorizations.append(authorization)
                if not authorization['authorized']:
                    control.contact_grants,control.allowed_pairs = previous_grants,previous_pairs
                    control.support_holds,control.support_events = previous_holds,[]
                    return {'admitted':False,'receipt':{'contact_authorizations':authorizations,
                                                       'authorization_transaction':'rolled_back'}}
        except Exception:
            control.contact_grants,control.allowed_pairs = previous_grants,previous_pairs
            control.support_holds,control.support_events = previous_holds,[]
            raise
        control._execution_horizon = horizon
        try:
            if op == 'move':
                targets = body['targets']
                from .motion_precision import tolerance, decorate as decorate_precision
                if 'position_tolerance_m' in body and physical is None:
                    raise ValueError('Explicit precision requires physical_contact_v1')
                control.position_tolerance_m = tolerance(body.get('position_tolerance_m', .003))
                push=body.get('push_contact')
                push_spec=None
                if push is not None:
                    if cloth is not None or marks or body.get('support_contact') is not None or body.get('tool_contact') is not None:
                        raise ValueError('Push contact requires a separate rigid non-grasp request')
                    if len(targets)!=1 or len(next(iter(targets.values())))!=1:
                        raise ValueError('Push contact requires one arm and no waypoints')
                    tokens=push.get('target_marks',[])
                    if not isinstance(tokens,list) or len(tokens)>8:
                        raise ValueError('Push contact accepts at most eight workpiece marks')
                    from .scene_geometry import array
                    def fresh_point(token):
                        mark=adapter._checked_marks.get(token)
                        if mark is None or mark['source_action_index']!=adapter._step_count:
                            raise ValueError('Push contact requires fresh visible surface marks')
                        return array(mark['xyz'])+array(control.native.robot_manager.scene.env_origins)[0]
                    push_spec={'body_point':fresh_point(push.get('body_mark')),
                        'surface_point':fresh_point(push.get('surface_mark')),
                        'target_points':[fresh_point(t) for t in tokens],
                        'fingers':push.get('fingers',[]),'patch_radius_m':push.get('patch_radius_m',.20)}
                support = body.get('support_contact')
                tool = body.get('tool_contact')
                if support is not None and tool is not None:
                    raise ValueError('Choose placement or tool stroke, not both in one move')
                contact = support if support is not None else tool
                support_point = None
                tool_points = None
                tool_handles = []
                retain_support = False
                if contact is not None:
                    if contact.get('mode') not in (('place',) if support is not None else ('stroke', 'reorient', 'adjust')):
                        raise ValueError('Unknown support contact mode')
                    mark=adapter._checked_marks.get(contact.get('surface_mark'))
                    if mark is None or mark['source_action_index']!=adapter._step_count:
                        raise ValueError('Support contact requires a fresh marked table surface')
                    from .scene_geometry import array
                    support_point=array(mark['xyz'])+array(control.native.robot_manager.scene.env_origins)[0]
                    if support is not None:
                        retain_support = support.get('retain_support',False)
                        if type(retain_support) is not bool:
                            raise ValueError('retain_support must be boolean')
                    else:
                        tokens=tool.get('target_marks',[])
                        tool_handles=tool.get('target_handles',[])
                        if (not isinstance(tokens,list) or not isinstance(tool_handles,list) or
                                any(not isinstance(h,str) or not h for h in [*tokens,*tool_handles]) or
                                not 0<=len(tokens)+len(tool_handles)<=8):
                            raise ValueError('Tool contact accepts at most eight fresh marks and prior workpiece references in total')
                        tool_points=[]
                        for token in tokens:
                            marked=adapter._checked_marks.get(token)
                            if marked is None or marked['source_action_index']!=adapter._step_count:
                                raise ValueError('Tool contact requires fresh marked workpiece surfaces')
                            tool_points.append(array(marked['xyz'])+array(control.native.robot_manager.scene.env_origins)[0])
                control.world.attachment_performance = None
                coordination=body.get('coordination','synchronized')
                if cloth is not None:
                    if support_point is not None:
                        raise ValueError('Cloth uses local finger contact and native table support; rigid tool/placement modes do not apply')
                    targets,coordination=cloth.prepare_move(targets,coordination)
                receipt = control.move(targets, timeout_s=360,
                                       max_actions=horizon.cap(80 if cloth else None) if horizon else (80 if cloth else None),
                                       coordination=coordination,push_contact=push_spec,
                                       support_point=support_point,retain_support=retain_support,tool_points=tool_points,
                                       tool_handles=tool_handles,
                                       tool_mode=tool.get('mode','stroke') if tool is not None else 'stroke',
                                       patch_radius_m=contact.get('patch_radius_m',.20) if contact else .20)
                if physical is not None:
                    decorate_precision(receipt, control, targets, control.position_tolerance_m)
                receipt['performance'] = {'attachment_views':control.world.attachment_performance,
                                          'geometry_policy':'immutable_shapes_reused; live_transforms_and_evidence_rechecked'}
                receipt['attachment_monitor'] = control.attachment_monitor
            elif op == 'wait':
                from .bounded_wait import wait
                receipt = wait(control, body['duration_s'])
            else:
                gripper_limit = 50 if body['openness'] <= .05 else 25
                receipt = control.gripper(body['role'], body['openness'],
                                          max_actions=horizon.cap(gripper_limit) if horizon else None)
            if horizon is not None:
                horizon.decorate(control, receipt)
            if physical is not None:
                receipt['physical_contact'] = physical.public()
                receipt['collision_guarantee'] = physical.public()['collision_guarantee']
                receipt['collision_policy'] = {**control.world.public_policy(), **physical.public()}
                if 'planning' in receipt:
                    receipt['planning']['collision_guarantee'] = physical.public()['collision_guarantee']
                if physical.recovery:
                    receipt.setdefault('host_adjustments',[]).append({'kind':'existing_contact_nonworsening_check','automatic_retreat':False})
            if cloth is not None:
                cloth.decorate(receipt,op)
            receipt.setdefault('host_adjustments',[]).extend(control.support_events)
            if hasattr(control, 'gripper_continuity'):
                receipt['host_adjustments'].extend(control.gripper_continuity.feedback())
            rates = list(control.motion_rate_events.values())
            receipt['host_adjustments'].extend(rates)
            receipt['host_adjustments'].extend(control.arm_continuity_events.values())
            if rates:
                receipt['joint_step_limit_rad'] = max(.08, receipt.get('joint_step_limit_rad', 0.))
                receipt['joint_step_limit_scope'] = 'maximum; per-role exceptions are reported in host_adjustments'
            receipt['active_support_holds']=[h.public() for h in control.support_holds.values()]
        except (ValueError,KeyError):
            # Admission errors must not silently revoke a previous contact
            # lease. Roll back only if no physics outcome became uncertain.
            if adapter._state_valid and adapter._step_count==body['action_index']:
                control.contact_grants,control.allowed_pairs=previous_grants,previous_pairs
                control.support_holds,control.support_events=previous_holds,[]
            raise
        finally:
            control._execution_horizon = None
        if len(authorizations)==1:
            receipt['contact_authorization'] = authorizations[0]
        elif authorizations:
            receipt['contact_authorizations'] = authorizations
    elif op != 'status':
        raise ValueError('Unknown configured operation')
    # Sensor packets remain public; opt-in checker receipts contain only reviewed semantic feedback.
    observation = copy.deepcopy(adapter._last_obs)
    observation['objects'] = []
    packet = EnvObservation.from_dict(observation).to_mcp_dict()
    for camera in packet['cameras']:
        camera.update(depth_encoding='uint16_png', depth_scale=1000.0)
    official = snapshot(adapter._env, reward_registered=adapter._reward_registered,
                        action_index=adapter._step_count)
    result = {'admitted': not (receipt and receipt.get('status') == 'rejected'),
        'observation': packet, 'info': {'official_outcome': official},
        'action_index': adapter._step_count, 'receipt': receipt,
        'control': {'controller_id': 'robodojo.checked_joint_ik.v1',
                    'layout_contract': getattr(adapter._env, '_openeta_layout_contract', None),
                    'sensor_contract': SENSOR_CONTRACT, 'roles': control.roles,
                    'completion_feedback':completion_profile,
                    'episode_budget':episode_budget(adapter),
                    'observation_clock':copy.deepcopy(getattr(adapter,'_observation_clock',{'status':'unavailable'})),
                    'articulations': {r['role']:{'id':r['role'],'joint_names':r['arm_joint_names']}
                                      for r in control.world.scene.snapshot['robots'] if r['role'] in control.roles},
                    'source_sha256': control.source_sha256,
                    'tcp_poses': {role: control.tcp(i).tolist() for i, role in enumerate(control.roles)},
                    'active_support_holds':[h.public() for h in control.support_holds.values()],
                    **({'physics_profile':material_profile} if material_profile else {}),
                    **({'initialization':initialization} if initialization else {}),
                    'collision_policy': {**control.world.public_policy(),
                        **(control.physical_contact.public() if getattr(control,'physical_contact',None) else {}),
                        **(cloth.policy() if cloth else {}),
                        **(control.fluid.policy() if getattr(control,'fluid',None) else {})}}}
    return result


def install_worker(worker):
    from starlette.routing import Route

    async def endpoint(request):
        handle = request.path_params['handle']
        env = worker._envs.get(handle)
        if env is None:
            return worker._json_response({'error': 'Unknown owned environment'}, 400)
        body = worker._safe_json_body(await request.body())

        def run():
            with worker._obs_lock_for(handle), worker._gl_lock:
                if worker._envs.get(handle) is not env:
                    raise ValueError('Environment closed before operation')
                adapter = env._env
                if getattr(adapter, '_backend', None) != 'robodojo':
                    raise ValueError('Configured endpoint requires RoboDojo')
                result = operate(adapter, body)
                worker._last_obs[handle] = adapter._last_obs
                if adapter._terminal:
                    worker._done_handles.add(handle)
                return result
        try:
            return worker._json_response(await worker._run_sim_call(run))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Details can contain private mesh paths: keep them in operator
            # logs. Host receives a meaningful category and state validity.
            from .public_errors import public_error_message
            adapter = getattr(env, '_env', None)
            return worker._json_response({'error': 'configured_operation_rejected',
                'error_type': type(exc).__name__,
                'state_valid': bool(getattr(adapter, '_state_valid', False)),
                'message': public_error_message(exc) if isinstance(exc, (ValueError, KeyError)) else 'See operator log'}, 400)

    worker.app.router.routes.append(Route('/env/{handle}/configured', endpoint, methods=['POST']))


def install_server(server):
    @server._blocking_tool
    @server._serialized_env_control
    def configured_robodojo(handle: str, request: dict, *, session_id: str = '') -> dict:
        """Private configured Host transport: calibrated status, marks and checked execution."""
        sid = session_id or server._current_session.get() or ''
        server._touch_session(sid)
        meta = server._session_envs.get(sid, {}).get(handle)
        if not meta or meta.get('backend') != 'robodojo':
            return {'error': 'Unknown owned RoboDojo environment'}
        return server._get_mgr().proxy_handle_op(meta, f'/env/{meta["remote_handle"]}/configured',
                                                method='POST', body=request)
