"""Private collision queries against one measured, thickened cloth surface.

A verdict is about this immutable frame only. It neither predicts cloth motion
nor authorizes execution. Native contact offset thickens the measured surface;
FCL surface distance is not a cloth penetration depth or contact force.
"""
import math
import time

import numpy as np

from .scene_geometry import forward_kinematics

from .deformable_surface import DeformableCoverageError


class ClothCollisionSnapshot:
    def __init__(self, frame, *, patches=None, permitted_finger_roles=()):
        import fcl
        self.fcl = fcl
        self.frame = frame
        self.patches = dict(patches or {})
        self.permitted_finger_roles = frozenset(permitted_finger_roles)
        self.objects = {}
        self.bounds = {}
        self.build_elapsed_s = 0.
        began = time.perf_counter()
        self._add(None, frame.triangles)
        for role, patch in self.patches.items():
            if patch.material_id is not frame.material_id or patch.source_revision > frame.revision:
                raise DeformableCoverageError('Cloth patch belongs to another material/frame')
            # Only triangles fully inside the selected material patch may
            # contact that hand's fingers. Boundary triangles remain checked.
            selected = np.isin(frame.triangles, patch.vertex_ids).all(axis=1)
            self._add(role, frame.triangles[~selected])
            self._add(("patch",role), frame.triangles[selected])
        self.build_elapsed_s = time.perf_counter()-began

    def _add(self, key, faces):
        if not len(faces):
            self.objects[key], self.bounds[key] = None, None
            return
        ids, local = np.unique(faces, return_inverse=True)
        vertices = self.frame.vertices[ids]
        shape = self.fcl.BVHModel()
        shape.beginModel(len(vertices), len(faces))
        shape.addSubModel(np.asarray(vertices, dtype=np.float64), local.reshape(-1,3).astype(np.int32))
        shape.endModel()
        self.objects[key] = self.fcl.CollisionObject(shape)
        self.bounds[key] = (vertices.min(axis=0), vertices.max(axis=0))

    def clearance(self, robot_object, *, role=None, finger=False):
        if finger and role in self.permitted_finger_roles:
            return math.inf
        key = role if finger and role in self.patches else None
        return self._clearance(robot_object,key)

    def patch_clearance(self, robot_object, role):
        return self._clearance(robot_object,("patch",role))

    def _clearance(self, robot_object, key):
        obj = self.objects[key]
        if obj is None:
            return math.inf
        hit = self.fcl.CollisionResult()
        self.fcl.collide(robot_object, obj, self.fcl.CollisionRequest(), hit)
        if hit.is_collision:
            return -self.frame.contact_offset_m
        value = float(self.fcl.distance(robot_object, obj, self.fcl.DistanceRequest(), self.fcl.DistanceResult()))
        if not math.isfinite(value) or value < 0:
            raise DeformableCoverageError('Measured cloth distance is unavailable')
        return value-self.frame.contact_offset_m

    def check(self, rigid_scene, positions, *, clearance_limit_m=.0002, allowed_world_ids=None, motion_bounds=None, part_floors=None):
        # Updating robot transforms uses the existing calibrated FK. Ignore
        # pairs only within this update call; no authorization is persisted.
        rigid_scene.check(positions, allowed_pairs=[(rigid_scene.items[i]['id'],rigid_scene.items[j]['id'])
                          for i,j in rigid_scene.pairs],clearance_limit=0.)
        if not math.isfinite(clearance_limit_m) or clearance_limit_m < 0:
            raise ValueError('Invalid measured cloth clearance threshold')
        frames = [forward_kinematics(r,q) for r,q in zip(rigid_scene.snapshot['robots'],positions)]
        minimum = math.inf
        contacts = []
        queries = 0
        for index, (item, obj) in enumerate(zip(rigid_scene.items, rigid_scene.objects)):
            world = item['robot_index'] is None
            if world and (allowed_world_ids is None or item['id'] in allowed_world_ids):
                continue
            role = None if world else rigid_scene.snapshot['robots'][item['robot_index']]['role']
            finger = not world and item['body'] in ('link7','link8')
            if finger and role in self.permitted_finger_roles:
                continue
            key = role if finger and role in self.patches else None
            box = self.bounds[key]
            if box is None:
                continue
            transform = rigid_scene.world_transforms[index] if world else frames[item['robot_index']][item['body']]
            points = rigid_scene.vertices[index]@transform[:3,:3].T+transform[:3,3]
            gap = np.maximum(np.maximum(points.min(axis=0)-box[1],box[0]-points.max(axis=0)),0.)
            bound = float(np.nextafter(np.linalg.norm(gap),0.))-self.frame.contact_offset_m-1e-10
            floor = (part_floors or {}).get(index, clearance_limit_m)
            threshold = .0002 if world else floor + (motion_bounds[index] if motion_bounds is not None else 0.)
            if bound > threshold:
                minimum = min(minimum,bound)
                continue
            clearance = self.clearance(obj, role=role, finger=finger)
            queries += 1
            minimum = min(minimum, clearance)
            if clearance <= threshold:
                contacts.append({'role':role,'body':item['body'],'part_index':index,'clearance_lower_bound_m':clearance,
                    'authorization_scope':'unapproved_world_cloth_contact' if world else
                    'outside_selected_material_patch' if finger and role in self.patches else 'unapproved_robot_cloth_contact'})
        return {'clear':not contacts,'clearance_lower_bound_m':minimum,'contacts':contacts,
                'queries':queries,'world_assumption':'measured_cloth_snapshot_only',
                'future_deformation_predicted':False,'cloth_force_measured':False}
