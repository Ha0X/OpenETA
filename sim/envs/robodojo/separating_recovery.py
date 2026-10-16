"""Bounded upward separation from an existing finger/table contact.

Other pairs retain the normal swept collision certificate. For each vertex
of an already touching finger, a derivative bound proves its height cannot
decrease along the proposed joint segment. Physics deviations are monitored
separately; no state teleport or collision flag change is used.
"""
import numpy as np

from .scene_geometry import array, forward_kinematics


def vertex_heights(scene, positions, indices):
    frames = [forward_kinematics(r,q) for r,q in zip(scene.snapshot['robots'],positions)]
    return {i: (scene.vertices[i] @ frames[scene.items[i]['robot_index']][scene.items[i]['body']][:3,:3].T
                + frames[scene.items[i]['robot_index']][scene.items[i]['body']][:3,3])[:,2]
            for i in indices}


def certify_upward(scene, start, target, indices, max_nodes=512):
    deltas = [dict(zip(r['joint_names'],array(b)-array(a))) for r,a,b in zip(scene.snapshot['robots'],start,target)]
    chains = {}
    acceleration = {}
    for i in indices:
        item = scene.items[i]
        r = scene.snapshot['robots'][item['robot_index']]
        parents = {j['child']:j for j in r['joints']}
        chain = []
        body = item['body']
        while body != r['base_link']:
            joint = parents[body]
            chain.append(joint)
            body = joint['parent']
        chains[i] = chain
        delta = deltas[item['robot_index']]
        angle = sum(abs(delta[j['name']]) for j in chain if j['kind']=='revolute')
        linear = sum(abs(delta[j['name']]) for j in chain if j['kind']=='prismatic')
        acceleration[i] = scene.bounds[item['robot_index']]*angle**2+2*angle*linear
    before = vertex_heights(scene,start,indices)
    after = vertex_heights(scene,target,indices)
    if min(float((after[i]-before[i]).min()) for i in indices) <= 1e-7:
        return {'status':'blocked','reason':'recovery_does_not_lift_each_contacting_finger_vertex'}
    pending = [(0.,1.)]
    nodes = 0
    while pending:
        if nodes >= max_nodes:
            return {'status':'unknown','reason':'separation_derivative_budget','samples':nodes}
        lo,hi = pending.pop()
        alpha = (lo+hi)/2
        positions = [array(a)+alpha*(array(b)-array(a)) for a,b in zip(start,target)]
        frames = [forward_kinematics(r,q) for r,q in zip(scene.snapshot['robots'],positions)]
        proved = True
        for i in indices:
            item = scene.items[i]
            index = item['robot_index']
            transform = frames[index][item['body']]
            points = scene.vertices[i] @ transform[:3,:3].T + transform[:3,3]
            velocity = np.zeros_like(points)
            for joint in chains[i]:
                if joint['kind']=='fixed':
                    continue
                t = frames[index][joint['parent']] @ array(joint['parent_frame'])
                axis = t[:3,:3] @ np.eye(3)['XYZ'.index(joint['axis'])]
                dq = deltas[index][joint['name']]
                velocity += (np.cross(axis,points-t[:3,3]) if joint['kind']=='revolute' else axis) * dq
            low_velocity = float(velocity[:,2].min())
            if low_velocity < -1e-9:
                return {'status':'blocked','reason':'recovery_would_deepen_finger_table_contact','samples':nodes+1}
            if low_velocity < acceleration[i]*(hi-lo)/2:
                proved = False
        nodes += 1
        if not proved:
            pending.extend([(lo,alpha),(alpha,hi)])
    return {'status':'clear','reason':'all_contacting_finger_vertices_monotonically_upward','samples':nodes}


class TableSeparation:
    def __init__(self, scene, positions, contacts, *, table):
        self.scene = scene
        ids = {item['id']:i for i,item in enumerate(scene.items)}
        self.pairs = []
        self.indices = set()
        for contact in contacts:
            a,b = contact['a'],contact['b']
            robot = b if a==table else a if b==table else None
            if robot is None:
                raise ValueError('Recovery only supports existing finger/table contacts')
            i = ids[robot]
            if scene.items[i]['robot_index'] is None or scene.items[i]['body'] not in ('link7','link8'):
                raise ValueError('Recovery requires finger geometry, not arm or camera contact')
            self.indices.add(i)
            self.pairs.append((a,b))
        if not self.indices:
            raise ValueError('No recoverable contact')
        self.heights = vertex_heights(scene,positions,self.indices)
        self.contacts = list(contacts)
        self.last_check = None

    def check_actual(self, positions):
        heights = vertex_heights(self.scene,positions,self.indices)
        ok = all(np.all(heights[i] >= self.heights[i]-1e-5) for i in self.indices)
        if not ok:
            distances = self.scene.signed_distances(positions, self.pairs)
            self.last_check = {'contacts': self.contacts, 'clearance_m': min(distances.values()),
                'minimum_vertex_height_change_m': min(float((heights[i]-self.heights[i]).min()) for i in self.indices),
                'nonworsening_tolerance_m': 1e-5}
        self.heights = {i:np.maximum(self.heights[i],heights[i]) for i in self.indices}
        return ok
