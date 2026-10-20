"""Fixed, hash-bound public workflow documents. No general filesystem access."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path

PHYSICAL_CONTRACT = 'references/contracts/physical-contact.md'
VISUAL_CLOTH_CONTRACT = 'references/contracts/cloth-visual.md'
PREFIX = 'plugins/openeta/skills/openeta-robodojo/'
BUNDLES = {
    'pick-place': ('references/pick-place.md', 'references/contracts/grasp.md', 'references/contracts/support.md'),
    'stack-support': ('references/stack-support.md', 'references/contracts/grasp.md', 'references/contracts/support.md'),
    'align-insert': ('references/align-insert.md', 'references/contracts/grasp.md', 'references/contracts/support.md'),
    'push-slide': ('references/push-slide.md', 'references/contracts/push.md'),
    'tool-use': ('references/tool-use.md', 'references/contracts/grasp.md', 'references/contracts/support.md', 'references/contracts/tool.md'),
    'articulated-interaction': ('references/articulated-interaction.md',),
    'cloth-contact': ('references/cloth-edge-grasp.md', 'references/contracts/cloth.md'),
    'moving-target': ('references/moving-target.md',),
}


class SkillLibrary:
    def __init__(self, source_sha256, *, root=None):
        root = Path(root or os.environ.get('OPENETA_ROBODOJO_SKILL_ROOT') or
                    Path(__file__).resolve().parents[1]/PREFIX).resolve()
        self._documents = {}
        for name in sorted({'SKILL.md', PHYSICAL_CONTRACT, VISUAL_CLOTH_CONTRACT, *(p for files in BUNDLES.values() for p in files)}):
            target = (root/name).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Skill reference escapes packaged root')
            raw = target.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            if not raw or len(raw) > 32768 or sha != source_sha256.get(PREFIX+name):
                raise ValueError('Skill document missing or differs from bound configuration: '+name)
            self._documents[name] = {'reference': name, 'sha256': sha, 'text': raw.decode('utf-8')}

    def read(self, skill_id, *, physical_contact=False, visual_cloth=False):
        if skill_id not in BUNDLES:
            raise ValueError('Unknown skill identifier')
        if physical_contact and not visual_cloth and skill_id == 'cloth-contact':
            raise ValueError('Cloth is unavailable in the rigid physical-contact profile')
        names = ((BUNDLES[skill_id][0], PHYSICAL_CONTRACT) if physical_contact else BUNDLES[skill_id])
        if visual_cloth:
            names = (BUNDLES[skill_id][0], PHYSICAL_CONTRACT, VISUAL_CLOTH_CONTRACT)
        return {
            'kind': 'packaged_skill_guidance', 'skill_id': skill_id,
            'physics_advanced': False, 'native_actions': 0,
            'consumes_tool_request': True, 'capabilities_changed': False,
            'scope': 'Public generic guidance only; not current scene evidence or contact authorization',
            'documents': [deepcopy(self._documents[name]) for name in names],
        }
