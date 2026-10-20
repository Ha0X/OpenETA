"""Explicit nonprehensile contact; never attach or reposition a pushed body."""
import numpy as np
from .scene_geometry import forward_kinematics
from .x5_calibration import link_to_tcp
from .tool_contact import marked_dynamic_body


class PushContact:
    max_penetration_m = .001

    def __init__(self, control, role, targets, *, body_point, surface_point,
                 target_points=(), fingers=('finger_1','finger_2'), patch_radius_m=.20):
        from .checked_control import pose_error
        if set(targets)!={role} or len(targets[role])!=1:
            raise ValueError('Push contact requires one arm, one short target and no waypoints')
        if (not isinstance(fingers,(list,tuple)) or not fingers or len(set(fingers))!=len(fingers)
                or not set(fingers)<={'finger_1','finger_2'}):
            raise ValueError('Select finger_1, finger_2 or both for push contact')
        if isinstance(patch_radius_m,bool) or not np.isfinite(patch_radius_m) or not .05<=patch_radius_m<=.35:
            raise ValueError('Push contact patch radius must be 0.05 to 0.35m')
        if len(target_points)>8:
            raise ValueError('Push contact accepts at most eight workpiece marks')
        control.world.refresh()
        self.role=role;self.index=control.roles.index(role);self.control=control
        current=control.tcp(self.index)
        self.requested=np.asarray(targets[role][0],dtype=float).copy()
        if (np.linalg.norm(self.requested[:3,3]-current[:3,3])>.05+1e-12
                or pose_error(current,self.requested)['orientation_rad']>.001):
            raise ValueError('Push contact requires at most 5cm translation with unchanged orientation')
        self.effective=self.requested.copy()
        scene=control.world.scene
        self.reference_tcp=link_to_tcp(forward_kinematics(scene.snapshot['robots'][self.index],
            control.world.positions()[self.index])['link6'])
        self.delta=self.requested[:3,3]-current[:3,3]
        self.point=np.asarray(surface_point,dtype=float);self.patch_radius_m=float(patch_radius_m)
        self.table=next(i for i,x in enumerate(scene.items) if x['id']==control.world.table_id)
        vertices=self.vertices(scene,self.table)
        lo,hi=vertices.min(axis=0),vertices.max(axis=0);self.top=float(hi[2])
        if (self.point.shape!=(3,) or not np.isfinite(self.point).all() or
                abs(self.point[2]-self.top)>.008 or np.any(self.point[:2]<lo[:2]) or np.any(self.point[:2]>hi[:2])):
            raise ValueError('Push surface mark must select the calibrated table top')
        self.root,indices=marked_dynamic_body(control.world,body_point)
        self.workpieces={self.root:indices}
        for point in target_points:
            root,parts=marked_dynamic_body(control.world,point)
            if root==self.root:
                raise ValueError('Pushed body and workpiece marks must differ')
            self.workpieces[root]=parts
        for owner,grant in control.contact_grants.items():
            if owner!=role and grant['root'] in self.workpieces:
                raise ValueError('Push contact cannot override another hand contact authorization')
            if owner==role and grant['root']!=self.root:
                raise ValueError('Release the previous contact authorization before selecting another pushed body')
        bodies={'finger_1':'link7','finger_2':'link8'}
        finger_indices=[i for i,x in enumerate(scene.items) if x['robot_index']==self.index
                        and x['body'] in {bodies[f] for f in fingers}]
        if {scene.items[i]['body'] for i in finger_indices}!={bodies[f] for f in fingers}:
            raise ValueError('Selected finger geometry unavailable')
        self.fingers=list(fingers)
        self.pairs=[(scene.items[i]['id'],scene.items[j]['id']) for i in finger_indices for j in indices]
        groups=list(self.workpieces.values())
        self.pairs.extend((scene.items[i]['id'],scene.items[j]['id'])
            for k,group in enumerate(groups) for other in groups[k+1:] for i in group for j in other)
        self.pairs.extend((scene.items[i]['id'],control.world.table_id) for group in groups for i in group)
        self.identities={i:scene.items[i]['id'] for i in [self.table,*finger_indices,
            *(i for group in groups for i in group)]}
        self.last=None
        # All selected bodies must begin near the support, not suspended loads.
        for group in groups:
            bottom=min(self.vertices(scene,i)[:,2].min() for i in group)
            if bottom-self.top>.01+1e-9:
                raise ValueError('Push contact requires initially table-supported bodies')
        result=self.check(scene,control.world.positions(),measured=True)
        if result['status']!='clear':
            raise ValueError('Initial push contact exceeds bounds: '+result['reason'])

    @staticmethod
    def vertices(scene,index):
        transform=scene.world_transforms[index]
        return scene.vertices[index] @ transform[:3,:3].T+transform[:3,3]

    def check(self,scene,positions,*,measured=False):
        from .checked_control import pose_error
        if any(i>=len(scene.items) or scene.items[i]['id']!=name for i,name in self.identities.items()):
            return {'status':'blocked','reason':'push_geometry_identity_changed'}
        pose=link_to_tcp(forward_kinematics(scene.snapshot['robots'][self.index],positions[self.index])['link6'])
        delta=pose[:3,3]-self.reference_tcp[:3,3]
        alpha=np.clip(float(delta@self.delta)/max(float(self.delta@self.delta),1e-12),0.,1.)
        if (np.linalg.norm(delta-alpha*self.delta)>.003 or
                pose_error(pose,self.reference_tcp)['orientation_rad']>.01):
            return {'status':'blocked','reason':'push_tcp_path_envelope_exceeded'}
        for group in self.workpieces.values():
            vertices=np.concatenate([self.vertices(scene,i) for i in group])
            if np.max(np.linalg.norm(vertices[:,:2]-self.point[:2],axis=1))>self.patch_radius_m:
                return {'status':'blocked','reason':'push_body_left_authorized_patch'}
            if self.top-vertices[:,2].min()>self.max_penetration_m:
                return {'status':'blocked','reason':'push_support_penetration_limit'}
        if measured:
            distances=scene.signed_distances(positions,self.pairs,contact_bounds=True,separation_bounds=True)
            depth=max(0.,-min(distances.values(),default=np.inf))
            self.last={'evaluation':'measured_collision_geometry','maximum_penetration_bound_m':float(depth),
                       'penetration_limit_m':self.max_penetration_m,
                       'distance_method':scene.last_contact_distance_method,
                       'force_or_contact_confirmed':False}
            if depth>self.max_penetration_m:
                return {'status':'blocked','reason':'push_contact_penetration_limit','measurement':self.last}
        return {'status':'clear','reason':'bounded_nonprehensile_contact'}

    def path_sampling_indices(self,scene):return range(len(scene.snapshot['robots']))
    def reached(self):return False
    def public(self):
        return {'mode':'nonprehensile_push','role':self.role,'fingers':self.fingers,
                'requested_tcp_xyz':self.requested[:3,3].tolist(),
                'effective_tcp_xyz':self.effective[:3,3].tolist(),
                'scope':'selected_fingers_to_pushed_body; selected_bodies_mutual_contact_and_table_only',
                'authorization_lifetime':'this_move_only','rigid_attachment_assumed':False,
                'gripper_command':'unchanged','max_translation_m':.05,'tcp_path_envelope_m':.003,
                'orientation_envelope_rad':.01,'measurement':self.last,
                'object_effect':'unverified; inspect the new RGB images',
                'guarantee':'unmarked_geometry_checked; moving_body_trajectory_not_predicted; per_physics_step_contact_bounds'}


def execute_push(control,targets,spec,*,timeout_s,max_actions):
    if len(targets)!=1 or getattr(control,'cloth',None) is not None:
        raise ValueError('Push contact requires one rigid-body arm request')
    role=next(iter(targets));placement=PushContact(control,role,targets,**spec)
    # Explicitly change the interaction model. Never restore a grasp assumption
    # after physical pushing; later grasping requires fresh authorization.
    revoked=control.contact_grants.pop(role,None) is not None
    control._expire_support([role],'nonprehensile_push_requested')
    control._rebuild_allowed_pairs()
    control.support_placement=placement
    control.attachment_monitor={'status':'nonprehensile_selected_body_not_attached','grasp_confirmed':False,
        'roles_with_evidence':[],'checked_physics_steps':0,'relative_motion':{}}
    try:
        from .motion_planning import execute_request
        receipt=execute_request(control,targets,timeout_s=timeout_s,max_actions=max_actions)
        for adjustment in receipt.get('host_adjustments',[]):
            if adjustment.get('kind')=='bounded_tool_contact_motion_rate':
                adjustment.update(kind='bounded_push_contact_motion_rate',scope='explicit_nonprehensile_contact',
                    safety_checks='per_physics_step_collision_contact_and_path_checks; pushed_body_not_attached')
        receipt['push_contact']=placement.public()
        receipt['collision_guarantee']=placement.public()['guarantee']
        receipt['planning']['collision_guarantee']=placement.public()['guarantee']
        receipt.setdefault('host_adjustments',[]).append({'kind':'explicit_nonprehensile_contact',
            'previous_grasp_authorization_revoked':revoked,**placement.public()})
        return receipt
    finally:
        control.support_placement=None
