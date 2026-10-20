"""Explicit, local held-object/table placement; never general contact disabling.

Only a short nearly vertical translation is admitted. The marked table patch,
held collider directory and maximum penetration are checked in planning and at
physical steps. Existing independent/shared-object coordination is not used.
"""
import numpy as np
from .scene_geometry import forward_kinematics


class SupportPlacement:
    max_penetration_m = .001
    patch_radius_m = .20

    def __init__(self, control, role, point, targets, *, patch_radius_m=.20):
        from .checked_control import pose_error
        from .attached_geometry import attached_scene
        from numbers import Real
        if (isinstance(patch_radius_m,bool) or not isinstance(patch_radius_m,Real) or
                not np.isfinite(patch_radius_m) or not .05<=patch_radius_m<=.35):
            raise ValueError('Support patch radius must be between 0.05m and 0.35m')
        self.patch_radius_m=float(patch_radius_m)
        self.bound_fallbacks=0
        if set(targets) != {role} or len(targets[role]) != 1:
            raise ValueError('Support placement requires one arm and one short target')
        control.world.refresh()
        self.control = control
        self.role = role
        self.index = control.roles.index(role)
        self.positions = control.world.positions()
        scene, evidence = attached_scene(control.world, self.positions, control.contact_grants)
        if (scene.snapshot.get('closed_chain_constraints') or not any(
                x['role']==role and x['mode']=='conditional_rigid_attachment' for x in evidence)):
            raise ValueError('Support placement requires measured two-finger evidence for one held body')
        self.indices = tuple(control.contact_grants[role]['indices'])
        self.table = next(i for i,x in enumerate(scene.items) if x['id']==control.world.table_id)
        self.pairs = [(scene.items[i]['id'],control.world.table_id) for i in self.indices]
        self.point = np.asarray(point,dtype=float)
        table = control.world.scene
        vertices = table.vertices[self.table] @ table.world_transforms[self.table][:3,:3].T + table.world_transforms[self.table][:3,3]
        lo,hi=vertices.min(axis=0),vertices.max(axis=0)
        if (self.point.shape!=(3,) or not np.isfinite(self.point).all() or
                abs(self.point[2]-hi[2])>.008 or np.any(self.point[:2]<lo[:2]) or np.any(self.point[:2]>hi[:2])):
            raise ValueError('Support mark must select the calibrated table top')
        self.top = float(hi[2])
        current=control.tcp(self.index);requested=np.asarray(targets[role][0],dtype=float)
        delta=requested[:3,3]-current[:3,3]
        rotation_error = pose_error(current,requested)['orientation_rad']
        # A rounded copy of the public current pose may differ by a fraction
        # of a micrometre. Treat only this near-zero upward request as a hold;
        # never execute it as an upward support motion.
        self.target_normalization = None
        requested_delta_z = float(delta[2])
        if -.05-1e-12 <= delta[2] < -.05:
            self.target_normalization = {'kind': 'support_descent_boundary_rounding',
                'requested_delta_z_m': requested_delta_z, 'effective_delta_z_m': -.05,
                'rounding_tolerance_m': 1e-12}
            delta[2] = -.05
        if 0 < delta[2] <= 1e-6:
            self.target_normalization = {'kind': 'near_zero_support_height',
                'requested_delta_z_m': float(delta[2]), 'effective_delta_z_m': 0.,
                'rounding_tolerance_m': 1e-6}
        if (np.linalg.norm(delta[:2])>.002 or not -.05<=delta[2]<=1e-6 or
                rotation_error>.001):
            raise ValueError('Support placement requires at most 5cm downward motion with unchanged orientation and XY; '
                f'requested delta_z_m={requested_delta_z:.17g}, xy_distance_m={np.linalg.norm(delta[:2]):.9g}, '
                f'orientation_error_rad={rotation_error:.9g}; limits: delta_z=[-0.05,0]m '
                '(up to 0.000001m rounded near-zero is held), xy=0.002m, orientation=0.001rad')
        self.requested = requested.copy()
        points=self.held_vertices(scene,self.positions)
        self._validate_patch(points)
        clearance=float(points[:,2].min()-self.top)
        if clearance < -self.max_penetration_m:
            raise ValueError('Existing support penetration exceeds placement limit')
        self.effective=requested.copy()
        # Authorised contact approach stops at the support instead of pursuing a
        # pose through it. Report this robot-target change in the receipt.
        self.effective[:3,:3]=current[:3,:3]
        self.effective[:2,3]=current[:2,3]
        self.effective[2,3]=current[2,3]+max(min(0.,delta[2]),-max(0.,clearance)-.0002)
        self.last=None
        self.check(control.world.scene,self.positions,measured=True)

    def held_vertices(self,scene,positions):
        frames=[forward_kinematics(r,q) for r,q in zip(scene.snapshot['robots'],positions)]
        points=[]
        for i in self.indices:
            item=scene.items[i]
            t=(scene.world_transforms[i] if item['robot_index'] is None else frames[item['robot_index']][item['body']])
            points.append(scene.vertices[i] @ t[:3,:3].T+t[:3,3])
        return np.concatenate(points)

    def _validate_patch(self,points):
        required=float(np.max(np.linalg.norm(points[:,:2]-self.point[:2],axis=1)))
        if required>self.patch_radius_m:
            raise ValueError(f'Held geometry leaves the marked support patch: required radius {required:.4f}m, authorized {self.patch_radius_m:.4f}m; choose a nearer visible table mark or explicitly request a radius up to 0.35m')

    def check(self,scene,positions,*,measured=False):
        points=self.held_vertices(scene,positions)
        try:self._validate_patch(points)
        except ValueError:return {'status':'blocked','reason':'support_patch_exceeded'}
        if not measured:
            # The calibrated table lies entirely below self.top. Translating
            # every held vertex above that plane is a sufficient separating
            # translation, hence a conservative penetration upper bound for
            # the finite table. Avoid repeated FCL queries at path samples.
            # This is not contact evidence; measured checks below remain intact.
            gap=float(points[:,2].min()-self.top)
            depth=max(0.,-gap)
            return {'status':'clear' if depth<=self.max_penetration_m else 'blocked',
                    'reason':'bounded_support_contact' if depth<=self.max_penetration_m else 'support_penetration_limit',
                    'measurement':{'table_plane_clearance_m':gap,'maximum_penetration_m':depth,
                        'evaluation':'predicted_geometry','penetration_limit_m':self.max_penetration_m,
                        'patch_radius_m':self.patch_radius_m,
                        'distance_method':'calibrated_table_halfspace_separation_upper_bound'}}
        distances=scene.signed_distances(positions,self.pairs,contact_bounds=measured)
        if measured and scene.last_contact_distance_method!='fcl_signed_distance':
            self.bound_fallbacks+=1
        clearance=min(distances.values())
        # Plane bound supplements convex signed distance at tabletop edges.
        depth=max(0.,-clearance,self.top-float(points[:,2].min()))
        measurement={'signed_clearance_m':clearance,'maximum_penetration_m':depth,
                   'evaluation': 'measured_geometry' if measured else 'predicted_geometry',
                   'penetration_limit_m':self.max_penetration_m,'patch_radius_m':self.patch_radius_m,
                   'distance_method':scene.last_contact_distance_method}
        if measured:self.last=measurement
        return {'status':'clear' if depth<=self.max_penetration_m else 'blocked',
                'reason':'bounded_support_contact' if depth<=self.max_penetration_m else 'support_penetration_limit',
                'measurement':measurement}

    def reached(self):
        return (self.last is not None and self.last['distance_method']=='fcl_signed_distance'
                and self.last['signed_clearance_m']<=.0003)

    def path_sampling_indices(self, scene):
        # Default conservatively includes all arms, including subclasses with
        # moving marked workpieces or future cross-arm constraints.
        return range(len(scene.snapshot['robots']))

    def public(self):
        return {'mode':'place','role':self.role,'scope':'held_body_to_marked_calibrated_table_patch_only',
                'requested_tcp_xyz':self.requested[:3,3].tolist(),'effective_tcp_xyz':self.effective[:3,3].tolist(),
                'target_normalization': self.target_normalization,
                'stop_on_contact':True,'contact_reached':self.reached(),'measurement':self.last,
                'conservative_bound_evaluations':self.bound_fallbacks,
                'guarantee':'sampled_contact_bounds_with_per_physics_step_monitor',
                'authorization_lifetime':'this_move_only','gripper_release':'agent_decision'}


class SupportContactScene:
    def __init__(self,scene,placement):self.scene=scene;self.placement=placement
    def __getattr__(self,name):return getattr(self.scene,name)
    def check(self,positions,*,allowed_pairs=(),**kwargs):
        result=self.scene.check(positions,allowed_pairs=[*allowed_pairs,*self.placement.pairs],**kwargs)
        support=self.placement.check(self.scene,positions)
        if support['status']!='clear':
            result['clear']=False
            result.setdefault('constraint_violations',[]).append(support)
        return result
    def check_path(self,start,target,*,allowed_pairs=(),timeout_s=15.,max_nodes=2048,**kwargs):
        import time
        began=time.monotonic()
        result=self.scene.check_path(start,target,allowed_pairs=[*allowed_pairs,*self.placement.pairs],
                                     timeout_s=timeout_s,max_nodes=max_nodes,**kwargs)
        if result['status']!='clear':return result
        indices = list(self.placement.path_sampling_indices(self.scene))
        extent = max((float(np.max(abs(np.asarray(target[i])-start[i]))) for i in indices), default=0.)
        count=max(2,int(np.ceil(extent/.002))+1)
        if count>max_nodes:return {'status':'unknown','reason':'support_contact_sample_budget'}
        for alpha in np.linspace(0,1,count):
            if time.monotonic()-began>timeout_s:return {'status':'unknown','reason':'support_contact_sample_budget'}
            q=[np.asarray(a)+(np.asarray(b)-a)*alpha for a,b in zip(start,target)]
            checked=self.placement.check(self.scene,q)
            if checked['status']!='clear':return checked
        return {**result,'support_contact_guarantee':'sampled_bounds',
                'support_samples': count, 'support_sample_articulations': indices}
