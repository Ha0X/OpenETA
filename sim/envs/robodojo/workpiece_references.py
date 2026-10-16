"""Episode-local identity references; never object pose or contact permission."""
from uuid import uuid4


class WorkpieceReferences:
    def __init__(self):
        self._entries = {}

    @staticmethod
    def _geometry(world, root, indices):
        try:
            if not indices or tuple(world.dynamic.get(root, ())) != tuple(indices):
                raise ValueError('Workpiece reference geometry is no longer available')
            return tuple(world.scene.items[i]['id'] for i in indices)
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError('Workpiece reference geometry is no longer available') from exc

    def issue(self, control, role, root, indices, source_action_index):
        grant = control.contact_grants.get(role)
        if grant is None or root == grant['root']:
            raise ValueError('Workpiece reference requires a distinct workpiece and current grasp authorization')
        indices = tuple(indices)
        geometry = self._geometry(control.world, root, indices)
        for token, entry in self._entries.items():
            if (entry['role'] == role and entry['grant'] is grant and entry['root'] == root
                    and entry['indices'] == indices and entry['geometry'] == geometry):
                return self._public(token, entry)
        token = 'workpiece_' + uuid4().hex
        entry = {'role': role, 'grant': grant, 'root': root, 'indices': indices,
                 'geometry': geometry, 'source_action_index': source_action_index}
        self._entries[token] = entry
        return self._public(token, entry)

    def resolve(self, control, role, token):
        entry = self._entries.get(token) if isinstance(token, str) else None
        if entry is None:
            raise ValueError('Unknown workpiece reference; use a fresh visible workpiece mark')
        if (entry['role'] != role or entry['grant'] is not control.contact_grants.get(role)):
            raise ValueError('Workpiece reference expired or belongs to another role/grasp authorization; use a fresh mark')
        if self._geometry(control.world, entry['root'], entry['indices']) != entry['geometry']:
            raise ValueError('Workpiece reference geometry identity changed; use a fresh mark')
        return entry['root'], entry['indices']

    @staticmethod
    def _public(token, entry):
        # Do not serialize the entry: it contains the private body directory.
        return {'handle': token, 'source_action_index': entry['source_action_index'],
                'scope': 'same_episode_role_and_grasp_authorization',
                'meaning': 'previously_agent_marked_identity_only; no_current_pose_or_contact_permission'}
