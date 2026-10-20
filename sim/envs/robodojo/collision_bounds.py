"""Conservative pairwise relative motion bounds for fixed articulated bodies.

A common upstream joint applies the same rigid transform to two links. It
cannot change their separation and is excluded from their relative bound.
"""
import numpy as np


class RelativeMotionBounds:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.chains = []
        for item in snapshot['colliders']:
            index = item['robot_index']
            if index is None:
                self.chains.append([])
                continue
            robot = snapshot['robots'][index]
            parents = {j['child']: j for j in robot['joints']}
            body = item['body']
            chain = []
            radius = float(np.linalg.norm(np.array(item['vertices']), axis=1).max())
            while body != robot['base_link']:
                joint = parents[body]
                # Child origin -> joint origin -> vertex, then walk upward.
                radius += float(np.linalg.norm(np.array(joint['child_frame'])[:3,3]))
                chain.append((joint['name'], joint['kind'], radius))
                radius += float(np.linalg.norm(np.array(joint['parent_frame'])[:3,3]))
                if joint['kind'] == 'prismatic':
                    lo, hi = robot['joint_limits'][robot['joint_names'].index(joint['name'])]
                    radius += max(abs(lo), abs(hi))
                body = joint['parent']
            self.chains.append(list(reversed(chain)))
        self._coefficient_pairs = None
        self._coefficients = None

    def _pair_coefficients(self, pairs):
        # Link chains and their conservative radii are immutable for this
        # snapshot. Cache only those coefficients, never poses or verdicts.
        key = tuple(pairs)
        if key == self._coefficient_pairs:
            return self._coefficients
        slots = {(i, name): k for k, (i, name) in enumerate(
            (i, name) for i, r in enumerate(self.snapshot['robots']) for name in r['joint_names'])}
        coefficients = np.zeros((len(pairs), len(slots)))
        angular = np.zeros_like(coefficients)
        items = self.snapshot['colliders']
        for row, (i,j) in enumerate(pairs):
            ia, ib = items[i]['robot_index'], items[j]['robot_index']
            a,b = self.chains[i], self.chains[j]
            common = 0
            if ia is not None and ia == ib:
                for left,right in zip(a,b):
                    if left[0] != right[0]:
                        break
                    common += 1
            for index, chain in [(ia,a[common:]),(ib,b[common:])]:
                if index is None:
                    continue
                for name,kind,radius in chain:
                    if kind == 'fixed':
                        continue
                    coefficients[row, slots[index,name]] += radius if kind=='revolute' else 1.
                    if kind == 'revolute':
                        angular[row, slots[index,name]] += 1.
        self._coefficient_pairs = key
        self._coefficients = coefficients
        self._angular_coefficients = angular
        return coefficients

    def for_segment(self, start, target, pairs):
        coefficients = self._pair_coefficients(pairs)
        deltas = np.asarray([abs(float(y)-float(x)) for a,b in zip(start,target) for x,y in zip(a,b)])
        # einsum avoids spawning BLAS workers for many small per-step queries.
        bounds = np.einsum('ij,j->i', coefficients, deltas)
        # A measured prismatic link can overtravel its nominal DOF limit.
        # Expand upstream lever-arm bounds for this actual segment; do not
        # clamp measured collision geometry or change physical/IK limits.
        expanded = []
        for r,a,b in zip(self.snapshot['robots'], start, target):
            kinds = {j['name']:j['kind'] for j in r['joints']}
            excess = sum(max(0., abs(float(a[k]))-max(abs(lo),abs(hi)),
                                      abs(float(b[k]))-max(abs(lo),abs(hi)))
                         for k,(name,(lo,hi)) in enumerate(zip(r['joint_names'],r['joint_limits']))
                         if kinds.get(name)=='prismatic')
            expanded.extend(excess*abs(float(y)-float(x)) for x,y in zip(a,b))
        if any(expanded):
            bounds += np.einsum('ij,j->i', self._angular_coefficients, np.asarray(expanded))
        bounds = np.where(bounds > 0, np.nextafter(bounds, np.inf), 0.)
        return {pair: float(bound) for pair,bound in zip(pairs,bounds)}
