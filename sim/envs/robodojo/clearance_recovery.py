"""Retreat from positive gaps below the normal path margin, without penetration.

The normal margin remains unchanged for every other pair. These initially
near pairs additionally receive a zero-margin swept certificate, sampled
nonworsening checks, and a strictly positive measured gap after each step.
"""
from .boundary_recovery import BoundaryRecovery

PATH_MARGIN_M = .0002


def near_clearance_pairs(scene, positions, allowed_pairs=()):
    # Fixed per-pair thresholds are essential: a shrinking nearest-distance
    # query can omit another pair that is also below the normal margin.
    result = scene.check(positions, allowed_pairs=allowed_pairs,
        clearance_limit=PATH_MARGIN_M,
        clearance_limits={pair: PATH_MARGIN_M for pair in scene.pairs})
    if not result['clear'] or result['clearance_m'] <= 0.:
        return [], result
    pairs = [(scene.items[i]['id'],scene.items[j]['id']) for i,j in result['uncertified_pairs']]
    return pairs, result


class ClearanceRecovery(BoundaryRecovery):
    def __init__(self, scene, positions, pairs, *, allowed_pairs=(), table_id=None):
        super().__init__(scene, positions, [{'a':a,'b':b} for a,b in pairs])
        if any(not 0 < d <= PATH_MARGIN_M+1e-9 for d in self.distances.values()):
            raise ValueError('Margin recovery requires positive, below-margin initial gaps')
        self.allowed_pairs=list(allowed_pairs)
        self.table_id=table_id
        self.plane_certificate_count=0
        self.mode='positive_clearance_retreat'
        self.failure_reason=None
        self.initial_minimum_gap_m=min(self.distances.values())
        ids={item['id']:item for item in scene.items}
        self.initial_clearance_check={'contacts':[{'a':a,'b':b,
            'robot_a':ids[a]['robot_index'],'robot_b':ids[b]['robot_index'],
            'measured_signed_clearance_m':self.distances[frozenset((a,b))]}
            for a,b in self.pairs],'clearance_m':self.initial_minimum_gap_m}

    def _separating_table_pairs(self, start, target):
        """Certify full convex solids above the calibrated horizontal table.

        The positive starting height plus the existing analytic nondecreasing
        vertex-height proof excludes intersection throughout the joint segment.
        It is not a contact permission, and does not replace measured checks.
        """
        if self.table_id is None:
            return []
        from .separating_recovery import vertex_heights, certify_upward
        import numpy as np
        ids={item['id']:i for i,item in enumerate(self.scene.items)}
        table=ids[self.table_id]
        if self.scene.items[table]['robot_index'] is not None:
            return []
        transform=self.scene.world_transforms[table]
        points=self.scene.vertices[table]@transform[:3,:3].T+transform[:3,3]
        top=float(points[:,2].max())
        pairs=[];indices=set()
        for a,b in self.pairs:
            other=b if a==self.table_id else a if b==self.table_id else None
            if other is not None and self.scene.items[ids[other]]['robot_index'] is not None:
                pairs.append((a,b));indices.add(ids[other])
        if not indices or any(float(h.min())<=top for h in vertex_heights(self.scene,start,indices).values()):
            return []
        proof=certify_upward(self.scene,start,target,indices)
        if proof['status']!='clear':
            return []
        self.plane_certificate_count+=1
        return pairs

    def check_segment(self, start, target):
        result=super().check_segment(start,target)
        if result['status']=='unknown':
            return result
        if result['status']!='clear':
            return {**result,'reason':'positive_gap_would_shrink'}
        # Do not ignore the near pair geometrically. Relax only its clearance
        # padding: its actual shapes still need a swept nonintersection proof.
        plane_pairs=self._separating_table_pairs(start,target)
        proof=self.scene.check_path(start,target,allowed_pairs=[*self.allowed_pairs,*plane_pairs],
            margin_m=0.,timeout_s=4.,max_nodes=1024)
        if proof['status']!='clear':
            return {'status':proof['status'],'reason':'positive_gap_swept_nonintersection_not_certified',
                    'detail':proof}
        return {**result,'reason':self.mode,
            'existing_contact_guarantee':'swept_nonintersection_plus_sampled_nonshrinking_gap',
            'separating_table_plane_pair_count':len(plane_pairs)}

    def check_actual(self, positions):
        ok=super().check_actual(positions)
        positive=self.last_check['clearance_m']>0.
        ok=ok and positive
        self.samples[-1]['accepted']=ok
        self.failure_reason=(None if ok else 'measured_positive_gap_not_preserved')
        return ok

    def diagnostics(self):
        return {**super().diagnostics(),'mode':self.mode,
            'initial_minimum_gap_m':self.initial_minimum_gap_m,
            'normal_path_margin_m':PATH_MARGIN_M,'actual_gap_must_remain_positive':True,
            'near_pair_path_guarantee':'zero_margin_swept_nonintersection_and_sampled_nonshrinking',
            'other_pair_path_guarantee':'unchanged_normal_margin',
            'separating_table_plane_certificates':self.plane_certificate_count,
            'failure_reason':self.failure_reason}


class MixedRecovery:
    """Compose existing-contact recovery with separate positive-gap checks.

    Exempting both groups from normal padding is valid only because each group
    retains its own checks. Existing overlaps are excluded from the positive
    group's zero-margin sweep, never from the original recovery monitor.
    """
    def __init__(self, existing, positive):
        self.existing = existing
        self.positive = positive
        self.pairs = [*existing.pairs, *positive.pairs]
        if len(set(map(frozenset, self.pairs))) != len(self.pairs):
            raise ValueError('Recovery groups must be disjoint')
        self.mode = 'mixed_contact_and_positive_clearance_retreat'
        self.initial_clearance_check = positive.initial_clearance_check
        self.last_check = None
        self.failure_reason = None

    @property
    def scene(self):
        return self.existing.scene

    @scene.setter
    def scene(self, value):
        self.existing.scene = value
        self.positive.scene = value

    def check_segment(self, start, target):
        if hasattr(self.existing, 'check_segment'):
            contact = self.existing.check_segment(start, target)
        else:
            from .separating_recovery import certify_upward
            contact = certify_upward(self.scene, start, target, self.existing.indices)
        if contact['status'] != 'clear':
            return {**contact, 'recovery_group': 'existing_contact'}
        positive = self.positive.check_segment(start, target)
        if positive['status'] != 'clear':
            return {**positive, 'recovery_group': 'initially_positive_gap'}
        return {'status': 'clear', 'reason': self.mode,
                'existing_contact_check': contact, 'positive_gap_check': positive}

    def check_actual(self, positions):
        # Evaluate both monitors even if the first fails, retaining each one's
        # measured history and original tolerances. Report the failing group.
        contact_ok = self.existing.check_actual(positions)
        positive_ok = self.positive.check_actual(positions)
        failed = self.existing if not contact_ok else self.positive
        self.last_check = failed.last_check
        self.failure_reason = (getattr(failed, 'failure_reason', None)
                               if not (contact_ok and positive_ok) else None)
        return contact_ok and positive_ok

    def diagnostics(self):
        return {'mode': self.mode,
                'existing_contact_mode': getattr(self.existing, 'mode', 'monotone_finger_table_separation'),
                'existing_contact': self.existing.diagnostics() if hasattr(self.existing, 'diagnostics') else
                    {'guarantee': 'monotone_vertex_height'},
                'positive_gap': self.positive.diagnostics(),
                'other_pair_path_guarantee': 'unchanged_normal_margin',
                'failure_reason': self.failure_reason}


def include_near_clearance(existing, scene, positions, *, allowed_pairs=(), table_id=None, actual_scene=None):
    """Add all positive near pairs not already owned by existing recovery."""
    excluded = [*allowed_pairs, *existing.pairs]
    pairs, _ = near_clearance_pairs(scene, positions, excluded)
    if not pairs:
        return existing
    positive = ClearanceRecovery(scene, positions, pairs,
        allowed_pairs=excluded, table_id=table_id)
    positive.actual_scene = actual_scene if actual_scene is not None else scene
    return MixedRecovery(existing, positive)
