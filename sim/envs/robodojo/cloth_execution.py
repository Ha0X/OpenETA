"""Short-horizon cloth execution: sampled deformation, scoped finger contact.

No particle coordinate, semantic landmark, or reward condition is public.
Native cloth physics owns deformation, table support, and self-contact.
"""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .deformable_collision import ClothCollisionSnapshot
from .deformable_surface import DeformableCoverageError, LocalPatchMonitor
from .native_cloth import NativeClothReader
from .scene_geometry import array

MODE = 'cloth_short_horizon_v1'
VISUAL_MODE = 'cloth_visual_short_horizon_v2'


class ClothExecution:
    def __init__(self, control, *, mode=MODE):
        if mode not in (MODE, VISUAL_MODE):
            raise ValueError("Unknown cloth execution mode")
        self.visual_only = mode == VISUAL_MODE
        self.control = control
        manager = control.native.scene_manager
        self.readers = [NativeClothReader(g, use_fabric=manager.use_fabric, simulation_time_s=0.)
                        for g in manager._garment_objects[0].values()]
        if not self.readers:
            raise DeformableCoverageError('Cloth profile has no physical garment')
        self.step = 0
        self.snapshots = []
        self.monitors = {}
        self.evidence = {}
        self.seen = set()
        self.lost_steps = {}
        self.events = []
        self.refresh()

    def grants(self):
        return {r:g for r,g in self.control.contact_grants.items() if g.get('kind')=='cloth_patch'}

    def refresh(self, *, advanced=False):
        self.step += int(advanced)
        seconds = self.step*float(self.control.native.sim.physics_dt)
        snapshots = []
        for index, reader in enumerate(self.readers):
            frame = reader.refresh(simulation_time_s=seconds)
            patches = {r:g['patch'] for r,g in self.grants().items() if g['cloth_index']==index}
            old = self.snapshots[index] if index<len(self.snapshots) else None
            same = old is not None and old.frame is frame and set(old.patches)==set(patches)
            if same:
                same = all(old.patches[r] is patch for r,patch in patches.items())
            snapshots.append(old if same else ClothCollisionSnapshot(frame, patches=patches,
                permitted_finger_roles=self.control.roles if getattr(self,"visual_only",False) else ()))
        self.snapshots = snapshots

    def authorize(self, role, point, *, contact_mode='surface', approach=None):
        if getattr(self,"visual_only",False):
            raise ValueError("Visual cloth profile does not accept material authorization")
        matches=[]
        for index,reader in enumerate(self.readers):
            # A calibrated image point identifies a surface, not an object pose.
            if np.linalg.norm(reader.surface.frame.vertices-point,axis=1).min() <= .011:
                matches.append(index)
        if not matches:return None
        if len(matches)!=1:
            return {'authorized':False,'reason':'ambiguous_cloth_surface'}
        index=matches[0]
        if contact_mode not in ('surface','local_layers'):
            raise ValueError('Unknown cloth contact mode')
        try:
            surface=self.readers[index].surface
            patch=(surface.local_layers(point,approach) if contact_mode=='local_layers'
                   else surface.material_patch(point,radius_m=.05))
        except DeformableCoverageError:
            return {'authorized':False,'reason':'ambiguous_or_insufficient_material_patch',
                    'guidance':'Mark an exposed cloth edge on a fresh image; overlapping layers cannot be silently selected.'}
        self.control.contact_grants[role]={'kind':'cloth_patch','cloth_index':index,'patch':patch,
                                           'root':f'cloth_{index}','indices':[],'pairs':[]}
        self.control._expire_support([role],'cloth_patch_reauthorization')
        self.control._rebuild_allowed_pairs()
        self.monitors.pop(role,None);self.seen.discard(role);self.lost_steps.pop(role,None)
        self.refresh()
        return {'authorized':True,'role':role,'scope':'selected_material_patch_to_selected_fingers_only',
                'cloth_contact':contact_mode,'material_ids_frozen':True,'new_layers_automatically_added':False,
                'local_layer_volume':({'radius_m':.05,'half_depth_m':.012,'axis':'requested_tcp_approach',
                                       'maximum_selected_area_m2':.025} if contact_mode=='local_layers' else None),
                'material_patch_radius_m':.05,'source':'agent_marked_calibrated_rgbd_surface',
                'grasp_confirmed':False,'rigid_attachment_created':False}

    def prepare_move(self, targets, coordination):
        from .motion_planning import validate_motion_targets
        validate_motion_targets(self.control,targets)
        held=getattr(self,"visual_only",False) or bool(self.grants())
        distance_limit, angle_limit=(.05,.25) if held else (.12,.5)
        factor=1.
        for role,path in targets.items():
            start=self.control.tcp(self.control.roles.index(role));goal=array(path[0])
            length=float(np.linalg.norm(goal[:3,3]-start[:3,3]))
            angle=float(Rotation.from_matrix(goal[:3,:3]@start[:3,:3].T).magnitude())
            factor=min(factor,distance_limit/max(length,1e-12),angle_limit/max(angle,1e-12))
        effective={}
        for role,path in targets.items():
            start=self.control.tcp(self.control.roles.index(role));goal=array(path[0]);pose=goal.copy()
            pose[:3,3]=start[:3,3]+factor*(goal[:3,3]-start[:3,3])
            pose[:3,:3]=Slerp([0.,1.],Rotation.from_matrix([start[:3,:3],goal[:3,:3]]))(factor).as_matrix()
            effective[role]=[pose]
        truncated=factor<1.-1e-12 or any(len(path)>1 for path in targets.values())
        self.events=[]
        if truncated:
            self.events.append({'kind':'cloth_short_horizon','requested_target_reached':False,
                'translation_limit_m':distance_limit,'rotation_limit_rad':angle_limit,
                'effective_targets':{r:p[0].tolist() for r,p in effective.items()},
                'remaining_waypoints_executed':False,
                'guidance':'Only this short segment will execute. Inspect the returned images before issuing the next segment toward your original goal.'})
        owners={g['cloth_index'] for r,g in self.grants().items()}
        shared=(len(targets)>1 if getattr(self,'visual_only',False) else len(self.grants())>len(owners))
        if shared and coordination!='synchronized':
            coordination='synchronized'
            self.events.append({'kind':'cloth_batch_synchronized' if getattr(self,'visual_only',False) else 'shared_cloth_synchronized',
                                'reason':'possible_shared_material_constraint; grasp_not_measured' if getattr(self,'visual_only',False) else 'independent_timing_could_change_material_tension',
                                'automatic_retiming_or_serialization':False})
        self.truncated=truncated
        return effective,coordination

    def check(self, positions, *, clearance_limit_m=.0002, motion_bounds=None, floors=None):
        results=[s.check(self.control.world.scene,positions,clearance_limit_m=clearance_limit_m,
                         allowed_world_ids=[self.control.world.table_id],motion_bounds=motion_bounds,
                         part_floors=(floors or {}).get(index)) for index,s in enumerate(self.snapshots)]
        return {'clear':all(r['clear'] for r in results),
                'clearance_lower_bound_m':min(r['clearance_lower_bound_m'] for r in results),
                'contacts':[{**c,'cloth_index':index} for index,r in enumerate(results) for c in r['contacts']],
                'world_assumption':'current_cloth_snapshot; no_future_deformation_prediction'}

    def check_segment(self, start, target):
        self.pending_recovery = None
        initial=self.check(start)
        if not initial['clear']:
            from .cloth_recovery import certify_recovery
            return certify_recovery(self,start,target,initial)
        return self._certify_segment(start,target)

    def _certify_segment(self, start, target, floors=None):
        # Certify robot motion against frozen cloth only. Live cloth is read
        # again after physics; the certificate does not predict its motion.
        scene=self.control.world.scene
        endpoint=self.check(target,floors=floors)
        if not endpoint['clear']:
            return {**endpoint,'reason':'predicted_cloth_clearance_at_command_target','samples':1,
                    'evidence':'predicted_robot_pose_against_current_cloth; not_measured_contact',
                    'guidance':('A nonworking robot part or world obstacle is too close to the cloth snapshot. Inspect the images and change your path; finger-cloth interaction is permitted.' if getattr(self,'visual_only',False) else 'The checked joint target approaches unapproved cloth material/parts. Inspect the reported hand and part; narrow the fingers, change the approach, or explicitly select a fresh exposed material patch. Other cloth layers are not implicitly authorized.')}
        began=time.monotonic();pending=[(start,target)];nodes=0;last=None
        while pending:
            if nodes>=128 or time.monotonic()-began>2.:
                return {'clear':False,'reason':'cloth_snapshot_check_budget','samples':nodes,
                        'evidence':'unresolved_clearance_certificate; not_a_measured_collision',
                        'last_uncertified':last,'guidance':'The check could not certify this segment within its budget. No collision is established by budget exhaustion; inspect the reported parts and request a smaller or more separated motion.'}
            a,b=pending.pop();bound=0.;part_bounds=[]
            for item,chain in zip(scene.items,scene.relative_bounds.chains):
                index=item['robot_index']
                if index is None:
                    part_bounds.append(0.);continue
                names=scene.snapshot['robots'][index]['joint_names']
                value=sum((radius if kind=='revolute' else 1.)*abs(b[index][names.index(name)]-a[index][names.index(name)])
                          for name,kind,radius in chain if kind!='fixed')
                value=float(np.nextafter(value,np.inf)) if value>0 else 0.
                part_bounds.append(value);bound=max(bound,value)
            result=self.check(a,motion_bounds=part_bounds,floors=floors);nodes+=1;last=result
            # Contacts here include pair gaps <= certificate threshold, not
            # necessarily an actual overlap. Never label them measured contact.
            if result['clear']:continue
            measured=self.check(a,floors=floors)
            if not measured['clear']:
                return {**measured,'reason':'cloth_snapshot_clearance','samples':nodes,
                        'evidence':'current_snapshot_clearance_below_required_margin; not_contact_force'}
            if bound<1e-9:
                return {**result,'clear':False,'reason':'cloth_snapshot_resolution','samples':nodes}
            middle=[(x+y)/2 for x,y in zip(a,b)]
            pending.extend([(middle,b),(a,middle)])
        return {'clear':True,'samples':nodes,'future_deformation_predicted':False}

    def check_after_physics(self, positions):
        result=self.check(positions)
        if getattr(self,'pending_recovery',None) is not None:
            from .cloth_recovery import check_measured_recovery
            return check_measured_recovery(self,positions,result)
        return result

    def measure(self):
        if getattr(self,"visual_only",False):
            # Private collision geometry is never a grasp sensor.
            return
        control=self.control
        q=control.world.positions()
        self.check(q)  # update measured robot collision transforms
        self.evidence={}
        for role,grant in self.grants().items():
            index=grant['cloth_index'];frame=self.readers[index].surface.frame
            wrist=control.tcp(control.roles.index(role)).copy()
            wrist[:3,3]+=array(control.native.robot_manager.scene.env_origins)[0]
            if role not in self.monitors or self.monitors[role].patch is not grant['patch']:
                self.monitors[role]=LocalPatchMonitor(grant['patch'],frame,wrist)
            distances=[]
            for link in ('link7','link8'):
                values=[self.snapshots[index].patch_clearance(obj,role) for item,obj in
                        zip(control.world.scene.items,control.world.scene.objects)
                        if item['robot_index']==control.roles.index(role) and item['body']==link]
                distances.append(min(values) if values else None)
            report=self.monitors[role].measure(frame,wrist,finger_clearances_m=distances)
            self.evidence[role]=report
            self._update_contact_evidence(role,report)
        for role in set(self.monitors)-set(self.grants()):
            self.monitors.pop(role,None);self.seen.discard(role);self.lost_steps.pop(role,None)

    def _update_contact_evidence(self, role, report):
        control=self.control
        closing_intent=control._grippers()[control.roles.index(role)]<=.05
        if not closing_intent:
            self.seen.discard(role);self.lost_steps.pop(role,None)
        elif report['proximity']=='both_fingers_near_patch':
            self.seen.add(role);self.lost_steps[role]=0
        elif role in self.seen:
            self.lost_steps[role]=self.lost_steps.get(role,0)+1
        report['holding_loss_monitor_armed']=closing_intent and role in self.seen
        report['relative_motion_reference']='first_sample_since_material_authorization; may_include_approach'
        if (control.operation=='move' and closing_intent
                and self.lost_steps.get(role,0)*float(control.native.sim.physics_dt)>=.04):
            control._halt('cloth_local_contact_evidence_lost',{'role':role,
                'guidance':'Two-finger proximity observed under a closed-gripper intent was subsequently lost. Inspect the images and regrasp; this is not proof of a drop.'})

    def decorate(self, receipt, operation):
        if operation=='move':
            receipt.setdefault('host_adjustments',[]).extend(self.events)
            if self.truncated:
                receipt['reached_execution_target']=receipt.get('reached_target',False)
                receipt['reached_target']=False
                receipt['continuation_required']=True
        if 'collision_policy' in receipt:
            receipt['collision_policy']={**receipt['collision_policy'],**self.policy()}
        if 'attachment_monitor' in receipt:
            receipt['attachment_monitor']['cloth_scope']='not_applicable; see cloth_monitor'
        if 'planning' in receipt:
            receipt['planning']['cloth_scope']='rigid_world_plan; cloth_snapshot_guard_before_each_physics_step'
        receipt['cloth_monitor']={'sampled_physics_steps':self.control.physics_steps,
            **({'grasp_state':'unknown; inspect returned images', 'mesh_grasp_inference':False}
               if getattr(self,'visual_only',False) else {'local_contacts':self.evidence}),
            'future_deformation_predicted':False,
            'rigid_attachment_used':False,'grasp_confirmed':False}
        if getattr(self,'recovery_events',None):
            receipt.setdefault('host_adjustments',[]).append({'kind':'cloth_positive_gap_recovery',
                'checked_physics_steps':len(self.recovery_events),'last':self.recovery_events[-1],
                'scope':'existing_positive_finger_gaps_only; no_new_contact; measured_nonworsening',
                'predicted_gap_tolerance_m':.000002,'physics_step_limit':40,
                'future_deformation_predicted':False})
        receipt['collision_guarantee']='rigid_robot_checks_plus_per_physics_step_cloth_snapshots; future_cloth_deformation_unknown'
        return receipt

    def policy(self):
        if getattr(self,'visual_only',False):
            return {'deformables':VISUAL_MODE,
                'supported_object_dynamics':'rigid_bodies_and_sampled_cloth',
                'world_pose_source':'live_rigid_poses_and_live_cloth_vertices; safety_only',
                'coverage':'protected_robot_pairs_swept; rigid_finger_contact_bounded; nonworking_parts_cloth_snapshots',
                'collision_guarantee':'rigid_robot_checks_plus_per_physics_step_cloth_snapshots; future_cloth_deformation_unknown',
                'cloth_contact_scope':'finger_links_to_cloth; table_support_and_self_contact_native',
                'cloth_collision_scope':'nonworking_robot_parts_and_other_world_geometry; private_safety_only',
                'cloth_future_deformation_prediction':False, 'cloth_batch_coordination':'synchronized',
                'cloth_motion_horizon':{'translation_m':.05,'rotation_rad':.25},
                'cloth_grasp_confirmed':False, 'cloth_force_or_tension_sensor':False,
                'cloth_mesh_grasp_inference':False}
        return {'deformables':MODE,'world_pose_source':'live_rigid_poses_and_live_cloth_vertices',
                'supported_object_dynamics':'rigid_bodies_and_sampled_cloth',
                'held_object_world_sweep':'rigid_only; cloth_checked_as_measured_deforming_surface',
                'cloth_contact_scope':'selected_local_material_to_selected_fingers; cloth_table_support_and_self_contact_native',
                'ik_admission':'rigid_scene_candidates; cloth_checked_before_each_physics_dispatch',
                'cloth_future_deformation_prediction':False,'cloth_shared_grasp_coordination':'synchronized',
                'cloth_motion_horizon':{'ungranted_translation_m':.12,'contact_translation_m':.05,
                                        'ungranted_rotation_rad':.5,'contact_rotation_rad':.25},
                'cloth_grasp_confirmed':False,'cloth_force_or_tension_sensor':False}
