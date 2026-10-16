"""Short upward lift from an already authorised held-body/table contact.

Predicted segments still separate monotonically. Actual compliant contact stays
inside the existing local support-contact penetration bound and must leave the
surface promptly. Live attachment/slip and other collision pairs still apply.
"""
import math
import numpy as np

from .boundary_recovery import BoundaryRecovery


class SupportSeparation(BoundaryRecovery):
    max_penetration_m = .001
    progress_window_s = .25
    max_contact_s = .5

    def __init__(self, control, scene, positions, contacts, attachments, targets):
        from .checked_control import pose_error
        held = [a for a in attachments if a['mode']=='conditional_rigid_attachment' and a['role'] in targets]
        if len(held)!=1 or scene.snapshot.get('closed_chain_constraints'):
            raise ValueError('Support separation requires one independently held body')
        role = held[0]['role']
        if set(targets)!={role} or len(targets[role])!=1:
            raise ValueError('Support separation requires one short single-arm lift')
        pose = control.tcp(control.roles.index(role)); target=np.asarray(targets[role][0])
        delta=target[:3,3]-pose[:3,3]
        if (np.linalg.norm(delta[:2])>.002 or not 0<delta[2]<=.05
                or pose_error(pose,target)['orientation_rad']>.025):
            raise ValueError('Support separation requires a vertical lift up to 5cm without rotation')
        body_ids={scene.items[i]['id'] for i in control.contact_grants[role]['indices']}
        table=control.world.table_id
        if not contacts or not all((c['a'] in body_ids and c['b']==table) or
                                    (c['b'] in body_ids and c['a']==table) for c in contacts):
            raise ValueError('Support separation only covers authorised held-body/table contacts')
        # Contact belongs to one rigid body, not an arbitrary subset of its
        # cooked convex parts. A part that is just above the table must not
        # retain ordinary path padding while its neighbour is separating.
        body_contacts=[{'a':scene.items[i]['id'],'b':table}
                       for i in control.contact_grants[role]['indices']]
        super().__init__(scene,positions,body_contacts)
        self.actual_scene=control.world.scene
        self.initial_distances=dict(self.distances)
        if min(self.distances.values()) < -self.max_penetration_m:
            raise ValueError('Existing support penetration exceeds 1mm bound')
        if min(self.distances.values()) > .0002:
            raise ValueError('Support separation requires overlap or a table gap within 0.2mm')
        self.initial_support_state='positive_near_gap' if min(self.distances.values())>0 else 'contact_or_overlap'
        self.mode='bounded_support_separation'
        self.period=float(control.native.sim.physics_dt)
        if not math.isfinite(self.period) or self.period<=0:
            raise ValueError('Unverified support separation timebase')
        self.progress_interval=max(1,math.ceil(self.progress_window_s/self.period))
        self.progress_baseline=dict(self.distances)
        self.failure_reason=None

    def check_segment(self, start, target):
        maximum=max(float(abs(np.asarray(b)-np.asarray(a)).max()) for a,b in zip(start,target))
        samples=max(2,int(np.ceil(maximum/.002))+1)
        if samples>65:
            return {'status':'unknown','reason':'recovery_sample_budget'}
        before=min(self.scene.signed_distances(start,self.pairs).values())
        previous=before
        for alpha in np.linspace(0.,1.,samples)[1:]:
            q=[np.asarray(a)+(np.asarray(b)-np.asarray(a))*alpha for a,b in zip(start,target)]
            current=min(self.scene.signed_distances(q,self.pairs).values())
            if current<previous-1e-7 or current<before-1e-7:
                return {'status':'blocked','reason':'existing_body_support_would_deepen'}
            previous=current
        if previous<=0 and previous<=before+1e-8:
            return {'status':'blocked','reason':'retreat_does_not_improve_body_support_clearance'}
        return {'status':'clear','reason':self.mode,'samples':samples,
                'existing_contact_guarantee':'discrete_nonworsening_whole_body_support_samples'}

    def check_actual(self, positions):
        # The parent records a bounded trace; its ratcheting verdict is replaced
        # ONLY for these admitted held-body/table pairs.
        super().check_actual(positions)
        current={frozenset((c['a'],c['b'])):c['measured_signed_clearance_m']
                 for c in self.last_check['contacts']}
        ok=all(d>=-self.max_penetration_m for d in current.values())
        self.failure_reason=None if ok else 'support_separation_envelope_exceeded'
        separated=all(d>.0002 for d in current.values())
        if ok and not separated and self.measured_steps%self.progress_interval==0:
            if min(current.values())<=min(self.progress_baseline.values())+.00002:
                ok=False; self.failure_reason='support_separation_no_measured_progress'
            self.progress_baseline=current.copy()
        if ok and not separated and self.measured_steps*self.period>=self.max_contact_s:
            ok=False; self.failure_reason='support_separation_contact_time_limit'
        self.last_check.pop('nonworsening_tolerance_m',None)
        self.last_check.update(support_penetration_limit_m=self.max_penetration_m,
            reference='absolute_support_envelope_with_progress_and_time_limits',
            maximum_deepening_from_initial_m=max(self.initial_distances[k]-d for k,d in current.items()))
        self.samples[-1].update(accepted=ok,maximum_deepening_from_initial_m=
            self.last_check['maximum_deepening_from_initial_m'])
        return ok

    def diagnostics(self):
        result=super().diagnostics()
        result.pop('nonworsening_tolerance_m',None)
        result={**result,'reference':'absolute_support_envelope_with_progress_and_time_limits',
                'support_penetration_limit_m':self.max_penetration_m,
                'progress_window_s':self.progress_window_s,'max_contact_s':self.max_contact_s,
                'elapsed_contact_sim_s':self.measured_steps*self.period,
                'failure_reason':self.failure_reason,'attachment_monitor':'live_each_physics_step'}
        result.update(initial_support_state=self.initial_support_state,
            initial_minimum_signed_clearance_m=min(self.initial_distances.values()),
            support_geometry_part_count=len(self.pairs),
            scope='one_authorized_held_rigid_body_to_calibrated_table',
            predicted_clearance='minimum_over_all_body_convex_parts',
            near_support_admission_m=.0002)
        return result
