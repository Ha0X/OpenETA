"""An explicit table support lease for a stationary independently held body."""
from copy import copy
import numpy as np

from .scene_geometry import forward_kinematics
from .support_contact import SupportPlacement


class SupportHold(SupportPlacement):
    def path_sampling_indices(self, scene):
        # Only this exact stationary lease has these dependencies. Unknown
        # subclasses retain the all-arm sampling policy. The enclosing scene
        # still certifies the complete joint path before these support samples.
        if type(self) is not SupportHold:
            return super().path_sampling_indices(scene)
        indices = {self.index}
        for i in (*self.indices, self.table):
            owner = scene.items[i]['robot_index']
            if owner is not None:
                indices.add(owner)
        return sorted(indices)

    @classmethod
    def from_placement(cls, placement):
        if not placement.reached():
            raise ValueError('Retained support requires measured placement contact')
        hold = cls.__new__(cls)
        hold.__dict__.update(copy(placement.__dict__))
        hold.root = placement.control.contact_grants[hold.role]['root']
        schema = hold.control.world.scene.snapshot['robots'][hold.index]
        hold.reference_wrist = forward_kinematics(schema, hold.control.world.positions()[hold.index])['link6']
        hold.release_in_progress = False
        return hold

    def check(self, scene, positions, *, measured=False, holding_evidence=None):
        from .checked_control import pose_error
        result = super().check(scene, positions, measured=measured)
        if result['status'] != 'clear':
            return result
        wrist = forward_kinematics(scene.snapshot['robots'][self.index],positions[self.index])['link6']
        drift = pose_error(wrist,self.reference_wrist)
        if drift['position_m'] > .002 or drift['orientation_rad'] > .02:
            return {'status':'blocked','reason':'stationary_support_owner_moved','owner_drift':drift}
        if measured and not self.release_in_progress:
            from .attached_geometry import attached_scene
            grant = self.control.contact_grants.get(self.role)
            if grant is None or grant['root'] != self.root:
                return {'status':'blocked','reason':'stationary_support_authorization_changed'}
            # The controller may supply evidence from its immediately preceding
            # pre-physics query. Never persist it: post-physics checks query anew.
            if holding_evidence is None:
                _, evidence = attached_scene(self.control.world,positions,self.control.contact_grants)
            else:
                evidence = holding_evidence
            if not any(x['role']==self.role and x['mode']=='conditional_rigid_attachment' for x in evidence):
                return {'status':'blocked','reason':'stationary_support_attachment_evidence_lost'}
        return result

    def public(self):
        return {'mode':'stationary_support','role':self.role,
                'scope':'held_body_to_original_marked_table_patch_only',
                'authorization_lifetime':'until_owner_move_or_measured_release_or_contact_reauthorization_or_episode_end',
                'owner_translation_limit_m':.002,'owner_orientation_limit_rad':.02,
                'measurement':self.last,'guarantee':'per_physics_step_measured_support_bounds'}
