"""LIBERO-style nonworsening collision recovery with explicit sampling scope.

New pairs retain swept clearance certification. Existing convex contacts are
checked at bounded joint samples and after every physical step. This is NOT
a continuous certificate for existing penetrating geometry.
"""
import numpy as np
from collections import deque

from .scene_geometry import array


class BoundaryRecovery:
    def __init__(self, scene, positions, contacts, *, release=False):
        self.scene = scene
        self.actual_scene = scene
        self.pairs = [(c['a'], c['b']) for c in contacts]
        if not self.pairs:
            raise ValueError('Recovery requires an existing collision')
        self.distances = scene.signed_distances(positions, self.pairs)
        self.release = release
        self.mode = 'bounded_release' if release else 'sampled_nonworsening_retreat'
        self.last_check = None
        self.samples = deque(maxlen=64)
        self.measured_steps = 0

    def check_segment(self, start, target):
        maximum = max(float(abs(array(b)-array(a)).max()) for a, b in zip(start, target))
        samples = max(2, int(np.ceil(maximum/.002))+1)
        if samples > 65:
            return {'status': 'unknown', 'reason': 'recovery_sample_budget'}
        before = self.scene.signed_distances(start, self.pairs)
        previous = before
        for alpha in np.linspace(0., 1., samples)[1:]:
            q = [array(a)+(array(b)-array(a))*alpha for a, b in zip(start, target)]
            current = self.scene.signed_distances(q, self.pairs)
            if any(current[k] < previous[k]-1e-7 or current[k] < before[k]-1e-7 for k in before):
                return {'status': 'blocked', 'reason': 'existing_contact_would_deepen'}
            previous = current
        progress = min(previous.values()) > min(before.values())+1e-8
        if not self.release and not progress and min(previous.values()) <= 0:
            return {'status': 'blocked', 'reason': 'retreat_does_not_improve_clearance'}
        return {'status': 'clear', 'reason': self.mode, 'samples': samples,
                'existing_contact_guarantee': 'discrete_nonworsening_samples'}

    def check_actual(self, positions):
        current = self.actual_scene.signed_distances(positions, self.pairs)
        ok = all(current[k] >= self.distances[k]-1e-5 for k in current)
        ids = {item['id']: item for item in self.actual_scene.items}
        self.last_check = {'contacts': [{'a': a, 'b': b, 'robot_a': ids[a]['robot_index'],
            'robot_b': ids[b]['robot_index'],
            'previous_signed_clearance_m': self.distances[frozenset((a,b))],
            'measured_signed_clearance_m': current[frozenset((a,b))]}
            for a,b in self.pairs], 'clearance_m': min(current.values()),
            'maximum_deepening_m': max(self.distances[k]-current[k] for k in current),
            'nonworsening_tolerance_m': 1e-5}
        self.measured_steps += 1
        self.samples.append({'step':self.measured_steps, 'minimum_signed_clearance_m':min(current.values()),
            'maximum_deepening_from_best_m':self.last_check['maximum_deepening_m'], 'accepted':ok})
        self.distances = {k: max(v, self.distances[k]) for k, v in current.items()}
        return ok

    def diagnostics(self):
        return {'reference':'best_measured_clearance_per_pair_since_recovery_start',
                'nonworsening_tolerance_m':1e-5, 'measured_steps':self.measured_steps,
                'recent_samples':list(self.samples), 'distance_source':'FCL_geometry_not_contact_force'}
