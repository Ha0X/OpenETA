"""Bounded joint-space search sharing the execution collision oracle.

Both arms are one search vector. Unselected arms and measured fingers are
fixed. Each accepted edge has the same swept clearance check as execution.
Finite failure returns unknown. No object pose or task answer leaves here.
"""
import time

import numpy as np

from .scene_geometry import array


class JointPlanner:
    def __init__(self, scene, selected, *, allowed_pairs=(), constraint=None, edge_timeout_s=2.):
        self.scene = scene
        self.selected = list(selected)
        self.allowed = allowed_pairs
        self.constraint = constraint
        if not np.isfinite(edge_timeout_s) or edge_timeout_s <= 0:
            raise ValueError('Edge check timeout must be finite and positive')
        self.edge_timeout_s = float(edge_timeout_s)
        self.slots = [(i, r['joint_names'].index(n)) for i, r in enumerate(scene.snapshot['robots'])
                      if i in self.selected for n in r['arm_joint_names']]
        self.limits = array([scene.snapshot['robots'][i]['joint_limits'][j] for i, j in self.slots])

    def plan(self, start, goals, *, timeout_s=15., max_nodes=256, seed=0):
        began = time.monotonic()
        deadline = began+timeout_s
        start = [array(q).copy() for q in start]
        def pack(q):
            return array([q[i][j] for i, j in self.slots])
        def unpack(v):
            q = [a.copy() for a in start]
            for (i, j), x in zip(self.slots, v):
                q[i][j] = x
            return q
        queries = 0
        last = None
        def edge(a, b):
            nonlocal queries, last
            if time.monotonic() >= deadline:
                return False
            qa, qb = unpack(a), unpack(b)
            if self.constraint is not None and not self.constraint(qa, qb):
                last = {'status': 'blocked', 'reason': 'agent_path_corridor'}
                return False
            last = self.scene.check_path(qa, qb, allowed_pairs=self.allowed,
                timeout_s=min(self.edge_timeout_s, max(0., deadline-time.monotonic())), max_nodes=512)
            queries += 1
            return last['status'] == 'clear'
        def receipt(path, mode, nodes):
            return path, {'status': 'clear' if path else 'unknown',
                'reason': mode, 'edge_queries': queries, 'search_nodes': nodes,
                'selected_articulations': self.selected, 'search_dof': len(self.slots),
                'elapsed_s': time.monotonic()-began, 'last_check': last,
                'edge_check_time_limit_s': self.edge_timeout_s,
                'world_assumption': 'stationary_snapshot'}
        root = pack(start)
        packed_goals = []
        for goal in goals:
            # Never silently discard an unselected-arm or finger target.
            v = pack(goal)
            if any(not np.allclose(a, b, atol=1e-9) for a, b in zip(unpack(v), goal)):
                raise ValueError('Joint planner goal changes a held joint')
            if edge(root, v):
                return receipt([start, unpack(v)], 'direct_joint_segment', 1)
            packed_goals.append(v)
        if not packed_goals:
            return receipt(None, 'no_admissible_ik_candidate', 0)
        rng = np.random.default_rng(seed)
        nodes, parents = [root], [-1]
        for count in range(max_nodes):
            if time.monotonic() >= deadline:
                break
            goal = packed_goals[count % len(packed_goals)]
            sample = goal if count % 4 == 0 else rng.uniform(self.limits[:, 0], self.limits[:, 1])
            near = int(np.argmin([np.linalg.norm(q-sample) for q in nodes]))
            delta = sample-nodes[near]
            candidate = nodes[near]+delta*min(1., .20/max(float(np.max(abs(delta))), 1e-12))
            if not edge(nodes[near], candidate):
                continue
            nodes.append(candidate); parents.append(near)
            if not edge(candidate, goal):
                continue
            path = [goal]
            idx = len(nodes)-1
            while idx >= 0:
                path.append(nodes[idx]); idx = parents[idx]
            path.reverse()
            # Greedy shortcutting only across independently certified edges.
            k = 0
            while k+2 < len(path) and time.monotonic() < deadline:
                if edge(path[k], path[k+2]):
                    path.pop(k+1)
                else:
                    k += 1
            return receipt([unpack(v) for v in path], 'joint_space_detour', len(nodes))
        return receipt(None, 'joint_path_search_budget_exhausted', len(nodes))
