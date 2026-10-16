"""Trusted, versioned harness presets. No simulator or model is started here."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'configs/harness'
KINDS = ('benchmark', 'embodiment', 'execution', 'interface')


class ConfigError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ConfigError('duplicate_config_key', f'Duplicate key: {key}')
        value[key] = item
    return value


def _read(path):
    try:
        return json.loads(path.read_text(), object_pairs_hook=_object,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError) as exc:
        raise ConfigError('invalid_config_file', f'{path.name}: {exc}') from exc


def _keys(value, expected, where):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ConfigError('invalid_config_schema', f'{where} requires exactly {sorted(expected)}')


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_]*_v[1-9][0-9]*', value):
        raise ConfigError('invalid_config_id', 'Use a registered, versioned config ID, not a path')
    return value


def _string(value, where, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ConfigError('invalid_config_schema', f'{where} must be a nonempty string')


FIELDS = {
    'benchmark': {'backend', 'env_pattern', 'outcome'},
    'embodiment': {'base', 'model', 'variant', 'groups', 'effectors', 'camera_policy'},
    'execution': {'controller_id', 'controller_option', 'backend', 'embodiments', 'scope',
                  'required_capabilities', 'implemented', 'missing_capabilities'},
    'interface': {'factory', 'tool_profile', 'effectors', 'supports_multiple_effectors',
                  'baseline_commit', 'source_sha256'},
}


def _validate_component(kind, id, component):
    optional = {'deformable_mode'} if kind=='execution' and isinstance(component,dict) and 'deformable_mode' in component else set()
    if kind == 'execution' and isinstance(component,dict) and 'contact_policy' in component:
        optional.add('contact_policy')
    if kind == 'execution' and isinstance(component, dict) and 'simulation_profile' in component:
        optional.add('simulation_profile')
    if kind == 'execution' and isinstance(component, dict) and 'physics_profile' in component:
        optional.add('physics_profile')
    if kind == 'benchmark' and isinstance(component, dict) and 'initialization_profile' in component:
        optional.add('initialization_profile')
    if kind == 'benchmark' and isinstance(component, dict) and 'episode_budget_override' in component:
        optional.add('episode_budget_override')
    if kind == 'benchmark' and isinstance(component, dict) and 'task_rule_supplement' in component:
        optional.add('task_rule_supplement')
    if kind == 'interface' and isinstance(component, dict) and 'visual_aids' in component:
        optional.add('visual_aids')
    if kind == 'interface' and isinstance(component, dict) and 'image_encoding' in component:
        optional.add('image_encoding')
    if kind == 'benchmark' and isinstance(component, dict) and 'completion_feedback' in component:
        optional.add('completion_feedback')
    if isinstance(component,dict):
        optional |= ({'native_layout','support_actors'} & component.keys()) if kind=='embodiment' else set()
        optional |= ({'support_policy'} & component.keys()) if kind=='execution' else set()
    _keys(component, FIELDS[kind] | {'id', 'kind', 'version'} | optional, id)
    _identifier(id)
    if component['id'] != id or component['kind'] != kind or type(component['version']) is not int or component['version'] < 1:
        raise ConfigError('invalid_component_identity', id)
    # Reject non-JSON / nonfinite payloads even for callers constructing catalogs.
    try:
        canonical(component)
    except (ValueError, TypeError) as exc:
        raise ConfigError('invalid_config_schema', f'{id}: nonfinite or non-JSON value') from exc
    if kind == 'benchmark':
        if 'task_rule_supplement' in component:
            from tools.task_rule_supplements import task_rule_supplement
            if component['task_rule_supplement'] is None:
                raise ConfigError('invalid_task_rule_supplement', id)
            try:
                task_rule_supplement(component, component.get('env_pattern'))
            except ValueError as exc:
                raise ConfigError('invalid_task_rule_supplement', id) from exc
        if 'completion_feedback' in component:
            profile = component['completion_feedback']
            if (component['backend'] != 'robodojo'
                    or profile not in ('build_tower_semantic_v1', 'robodojo_native_coarse_v1')
                    or (profile == 'build_tower_semantic_v1'
                        and component['env_pattern'] != 'openeta/robodojo_dual_x5_build_tower-v0')):
                raise ConfigError('invalid_completion_feedback', id)
        if 'initialization_profile' in component:
            if (component['initialization_profile'] != 'egg_precheck_settle_300_v1'
                    or component['backend'] != 'robodojo'
                    or component['env_pattern'] != 'openeta/robodojo_dual_x5_fill_egg_holder-v0'):
                raise ConfigError('invalid_initialization_profile', id)
        if 'episode_budget_override' in component:
            tasks = {'align_blocks_400_v1': 'align_blocks', 'build_tower_2000_v1': 'build_tower', 'build_tower_4000_v1': 'build_tower',
                     'play_tic_tac_toe_native_adapter_v101': 'play_tic_tac_toe',
                     'play_tic_tac_toe_1300_v104': 'play_tic_tac_toe',
                     'imitate_sorting_sequence_native_adapter_v101': 'imitate_sorting_sequence'}
            task = tasks.get(component['episode_budget_override'])
            unlimited = component['episode_budget_override'] == 'robodojo_unlimited_actions_v105'
            if unlimited and component['backend'] != 'robodojo':
                raise ConfigError('invalid_episode_budget_override', id)
            if not unlimited and (task is None or component['backend'] != 'robodojo'
                    or component['env_pattern'] != f'openeta/robodojo_dual_x5_{task}-v0'):
                raise ConfigError('invalid_episode_budget_override', id)
        for key in ('backend', 'env_pattern', 'outcome'):
            if not isinstance(component[key], str) or not component[key]:
                raise ConfigError('invalid_benchmark', key)
        try:re.compile(component['env_pattern'])
        except re.error as exc:raise ConfigError('invalid_env_pattern', str(exc)) from exc
    elif kind == 'embodiment':
        for key in ('base', 'model', 'variant', 'camera_policy'):
            _string(component[key], key)
        groups, effectors = component['groups'], component['effectors']
        if not isinstance(groups, dict) or not groups or not isinstance(effectors, dict) or not effectors:
            raise ConfigError('invalid_embodiment', 'Nonempty groups and effectors are required')
        for name, group in groups.items():
            _string(name, 'group name')
            _keys(group, {'kind', 'joint_names'} | ({'articulation'} if 'articulation' in group else set()), name)
            if 'articulation' in group:
                _string(group['articulation'], 'articulation')
            _string(group['kind'], 'group kind')
            joints = group['joint_names']
            if joints is not None and (not isinstance(joints, list) or not joints or
                    any(not isinstance(j, str) or not j for j in joints) or len(set(joints)) != len(joints)):
                raise ConfigError('invalid_joint_names', name)
        for name, effector in effectors.items():
            _string(name, 'effector name')
            _keys(effector, {'group', 'kind', 'calibration', 'reported_position_frame',
                            'reported_orientation_frame', 'tcp_rotation_from_reported', 'max_aperture_m',
                            'affected_by_groups'} | ({'tcp_translation_from_reported'} if 'tcp_translation_from_reported' in effector else set()), name)
            if 'tcp_translation_from_reported' in effector:
                import numpy as np
                translation = np.asarray(effector['tcp_translation_from_reported'], dtype=float)
                if translation.shape != (3,) or not np.isfinite(translation).all():
                    raise ConfigError('invalid_tcp_translation', name)
            _string(effector['group'], 'effector group')
            _string(effector['kind'], 'effector kind')
            for key in ('calibration', 'reported_position_frame', 'reported_orientation_frame'):
                _string(effector[key], key, nullable=True)
            if effector['group'] not in groups:
                raise ConfigError('unknown_control_group', name)
            affected = effector['affected_by_groups']
            if (not isinstance(affected, list) or any(not isinstance(g, str) for g in affected)
                    or len(set(affected)) != len(affected) or effector['group'] not in affected
                    or set(affected) - set(groups)):
                raise ConfigError('invalid_effector_dependencies', name)
            rotation = effector['tcp_rotation_from_reported']
            if rotation is not None:
                import numpy as np
                try:
                    r = np.asarray(rotation, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ConfigError('invalid_tcp_rotation', name) from exc
                if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(r), 1):
                    raise ConfigError('invalid_tcp_rotation', name)
            aperture = effector['max_aperture_m']
            if aperture is not None and (type(aperture) not in (int, float) or aperture <= 0):
                raise ConfigError('invalid_effector_aperture', name)
    elif kind == 'execution':
        if 'physics_profile' in component and (component['physics_profile'] != 'cloth_friction_2x_v1' or component.get('deformable_mode') != 'cloth_visual_short_horizon_v2'):
            raise ConfigError('invalid_execution', 'unsupported physics_profile')
        if 'simulation_profile' in component and (component['simulation_profile'] != 'gpu_tensor_v1' or component.get('backend') != 'robodojo'):
            raise ConfigError('invalid_execution', 'Unsupported simulation profile')
        if 'contact_policy' in component and (component['contact_policy'] != 'physical_contact_v1' or component.get('backend') != 'robodojo' or component.get('deformable_mode') not in (None,'cloth_visual_short_horizon_v2')):
            raise ConfigError('invalid_execution', 'unsupported contact_policy or cloth combination')
        if 'deformable_mode' in component and (component['deformable_mode'] not in ('cloth_short_horizon_v1','cloth_visual_short_horizon_v2') or component.get('backend') != 'robodojo'):
            raise ConfigError('invalid_execution', 'unsupported deformable_mode')
        if component.get('deformable_mode') == 'cloth_visual_short_horizon_v2' and component.get('contact_policy') != 'physical_contact_v1':
            raise ConfigError('invalid_execution', 'visual cloth requires physical_contact_v1')
        for key in ('controller_id', 'backend'):
            _string(component[key], key)
        _string(component['controller_option'], 'controller_option', nullable=True)
        if type(component['implemented']) is not bool:
            raise ConfigError('invalid_execution', 'implemented must be boolean')
        for key in ('embodiments', 'scope', 'missing_capabilities'):
            if not isinstance(component[key], list) or any(not isinstance(x, str) or not x for x in component[key]):
                raise ConfigError('invalid_execution', key)
        if not component['scope'] or len(set(component['scope'])) != len(component['scope']):
            raise ConfigError('invalid_execution', 'scope must contain unique groups')
        if not isinstance(component['required_capabilities'], dict):
            raise ConfigError('invalid_execution', 'required_capabilities must be an object')
    elif kind == 'interface':
        if 'image_encoding' in component and (component['image_encoding'] != 'lossless_webp_if_smaller_v1'
                or component.get('factory') != 'configured_robodojo_atomic'):
            raise ConfigError('invalid_image_encoding', id)
        if 'visual_aids' in component and (component['visual_aids'] != 'tcp_overlay_v1'
                or component.get('factory') != 'configured_robodojo_atomic'):
            raise ConfigError('invalid_visual_aids', id)
        for key in ('factory', 'tool_profile'):
            _string(component[key], key)
        _string(component['baseline_commit'], 'baseline_commit', nullable=True)
        if type(component['supports_multiple_effectors']) is not bool or not isinstance(component['effectors'], list):
            raise ConfigError('invalid_interface', id)
        if not isinstance(component['source_sha256'], dict):
            raise ConfigError('invalid_interface', 'source_sha256 must be an object')
        if not component['effectors'] or any(not isinstance(x, str) or not x for x in component['effectors']):
            raise ConfigError('invalid_interface', 'effectors must contain kind names')
        for path, sha in component['source_sha256'].items():
            if not path or Path(path).is_absolute() or '..' in Path(path).parts or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
                raise ConfigError('invalid_source_fingerprint', path)


@dataclass(frozen=True)
class ResolvedConfig:
    """Canonical immutable payload; returned dictionaries are independent copies."""
    encoded: str

    @property
    def value(self):return json.loads(self.encoded)

    @property
    def id(self):return self.value['id']

    @property
    def sha256(self):return hashlib.sha256(self.encoded.encode()).hexdigest()

    def public(self):
        c = self.value['components']
        return {'id': self.id, 'sha256': self.sha256, 'description': self.value['description'],
                'backend': c['benchmark']['backend'], 'robot_model': c['embodiment']['model'],
                'robot_variant': c['embodiment']['variant'],
                'effectors': {name: {'kind': item['kind'], 'group': item['group']}
                              for name, item in c['embodiment']['effectors'].items()},
                'execution_implemented': c['execution']['implemented'],
                'missing_capabilities': c['execution']['missing_capabilities'],
                'runtime_validation_required': True}

    def require_executable(self, *, source_root=ROOT):
        c = self.value['components']
        if not c['execution']['implemented'] or c['execution']['missing_capabilities']:
            raise ConfigError('config_not_executable', f'{self.id}: missing {c["execution"]["missing_capabilities"]}')
        # Factories are Python-owned dispatch, never dynamic imports from JSON.
        if c['interface']['factory'] == 'configured_robodojo_atomic':
            self._require_robodojo(source_root)
            return
        if c['interface']['factory'] != 'legacy_libero_atomic':
            raise ConfigError('unsupported_interface_factory', c['interface']['factory'])
        # This adapter preserves the old implementation; it cannot apply an
        # arbitrary new calibration just because a JSON catalog parses.
        from tools.codex_atomic_geometry import PANDA_BODY_TO_GRIP_SITE
        body = c['embodiment']
        primary = body['effectors'].get('primary', {})
        if (body['model'] != 'panda' or body['base'] != 'fixed' or body['variant'] != 'single'
                or set(body['groups']) != {'arm'} or set(body['effectors']) != {'primary'}
                or c['execution']['scope'] != ['arm'] or c['benchmark']['backend'] != 'libero'
                or c['execution']['controller_id'] != 'mink.robosuite_joint_velocity'
                or c['execution']['controller_option'] != 'mink_joint_velocity'
                or c['interface']['tool_profile'] != 'atomic'
                or primary.get('group') != 'arm' or primary.get('kind') != 'parallel_gripper'
                or primary.get('affected_by_groups') != ['arm']
                or primary.get('reported_position_frame') != 'grip_site'
                or primary.get('reported_orientation_frame') != 'right_hand'
                or primary.get('tcp_rotation_from_reported') != PANDA_BODY_TO_GRIP_SITE.tolist()):
            raise ConfigError('unsupported_legacy_configuration', 'Legacy LIBERO adapter requires its measured Panda geometry and arm-only Mink scope')
        changed = [p for p, sha in c['interface']['source_sha256'].items()
                   if not (source_root / p).is_file() or hashlib.sha256((source_root / p).read_bytes()).hexdigest() != sha]
        if changed:
            raise ConfigError('config_source_mismatch', f'Pinned implementation changed: {changed}')

    def _require_robodojo(self, source_root):
        from sim.envs.robodojo.x5_calibration import TCP_FROM_LINK, ROBOT_CONTRACT
        c = self.value['components']
        body = c['embodiment']
        roles = {'primary':'arm'} if body['variant']=='single_x5' else {'left':'left_arm','right':'right_arm'}
        if (body['base'] != 'fixed' or body['model'] != 'x5' or body['variant'] not in ('single_x5','dual_x5')
                or set(body['groups']) != set(roles.values()) or set(body['effectors']) != set(roles)
                or c['benchmark']['backend'] != 'robodojo'
                or c['execution']['controller_id'] != 'robodojo.checked_joint_ik.v1'
                or c['execution']['controller_option'] != 'robodojo_checked'
                or c['execution']['scope'] != list(roles.values())):
            raise ConfigError('unsupported_robodojo_configuration', 'Fixed X5 calibrated profile required')
        from sim.envs.robodojo.layout import COMPETITION, SUPPORT_POLICY
        support = body.get('support_actors', {})
        policy = c['execution'].get('support_policy')
        if support or policy or 'native_layout' in body:
            expected = {'support_arm0':{'model':'franka','control_mode':'task_managed','agent_controllable':False}}
            if (support != expected or policy != SUPPORT_POLICY or body.get('native_layout') != COMPETITION
                    or body['variant'] != 'dual_x5' or c['execution'].get('contact_policy') != 'physical_contact_v1'):
                raise ConfigError('unsupported_robodojo_configuration', 'Explicit native auxiliary ownership contract required')
        for role,group in roles.items():
            effector = body['effectors'][role]
            if (effector.get('group') != group or effector.get('kind') != 'parallel_gripper'
                    or effector.get('calibration') != ROBOT_CONTRACT
                    or effector.get('tcp_rotation_from_reported') != TCP_FROM_LINK[:3,:3].tolist()
                    or effector.get('tcp_translation_from_reported') != TCP_FROM_LINK[:3,3].tolist()
                    or body['groups'][group]['joint_names'] != [f'joint{i}' for i in range(1,7)]
                    or (len(roles)>1 and body['groups'][group].get('articulation') != role)):
                raise ConfigError('unsupported_robodojo_configuration', 'X5 role/joint/TCP binding differs from the verified profile')
        changed = [p for p, sha in c['interface']['source_sha256'].items()
                   if not (source_root/p).is_file() or hashlib.sha256((source_root/p).read_bytes()).hexdigest() != sha]
        if changed:
            raise ConfigError('config_source_mismatch', f'Pinned implementation changed: {changed}')

    def bind_task(self, *, env_id, task, seed):
        c = self.value['components']
        if not isinstance(env_id, str) or not re.fullmatch(c['benchmark']['env_pattern'], env_id):
            raise ConfigError('config_task_mismatch', f'{self.id} does not support environment {env_id!r}')
        if c['benchmark']['backend'] == 'robodojo' and f'robodojo_{c["embodiment"]["variant"]}_' not in env_id:
            raise ConfigError('config_embodiment_mismatch', 'Environment robot variant differs from preset')
        if not isinstance(task, str) or not task.strip() or type(seed) is not int:
            raise ConfigError('invalid_task_instance', 'Nonempty task text and integer seed required')
        value = {'schema_version': 'openeta.harness_binding.v1', 'config_id': self.id,
                 'config_sha256': self.sha256, 'task_instance': {'env_id': env_id, 'task': task, 'seed': seed}}
        return {**value, 'binding_sha256': digest(value)}


class ConfigRegistry:
    def __init__(self, directory=CATALOG):
        components, presets = _read(directory / 'components.json'), _read(directory / 'presets.json')
        _keys(components, {'schema_version', 'components'}, 'components file')
        _keys(presets, {'schema_version', 'presets'}, 'presets file')
        if type(components['schema_version']) is not int or components['schema_version'] != 1 or type(presets['schema_version']) is not int or presets['schema_version'] != 1:
            raise ConfigError('unsupported_config_version', 'Expected schema_version=1')
        catalog = components['components']
        _keys(catalog, KINDS, 'component kinds')
        for kind, entries in catalog.items():
            if not isinstance(entries, dict):raise ConfigError('invalid_components', kind)
            for id, value in entries.items():_validate_component(kind, id, value)
        self._resolved = {}
        if not isinstance(presets['presets'], list):raise ConfigError('invalid_presets', 'presets must be an array')
        for preset in presets['presets']:
            _keys(preset, {'id', 'description', 'components'}, 'preset')
            id = _identifier(preset['id'])
            _string(preset['description'], 'preset description')
            if id in self._resolved:raise ConfigError('duplicate_config_id', id)
            _keys(preset['components'], KINDS, id)
            try:resolved = {kind: catalog[kind][preset['components'][kind]] for kind in KINDS}
            except (KeyError, TypeError) as exc:raise ConfigError('missing_component', str(exc)) from exc
            b, e, x, i = (resolved[k] for k in KINDS)
            if x['backend'] != b['backend'] or e['id'] not in x['embodiments']:
                raise ConfigError('incompatible_components', id)
            if set(x['scope']) - set(e['groups']):raise ConfigError('unknown_control_group', id)
            if len(e['effectors']) > 1 and not i['supports_multiple_effectors']:
                raise ConfigError('incompatible_interface', id)
            if any(v['kind'] not in i['effectors'] for v in e['effectors'].values()):
                raise ConfigError('incompatible_effector', id)
            if x['implemented']:
                if x['missing_capabilities'] or any(g['joint_names'] is None for g in e['groups'].values()):
                    raise ConfigError('incomplete_executable_config', id)
                if any(any(v[k] is None for k in ('calibration', 'reported_position_frame',
                       'reported_orientation_frame', 'tcp_rotation_from_reported')) for v in e['effectors'].values()):
                    raise ConfigError('incomplete_executable_config', id)
                if not i['source_sha256']:
                    raise ConfigError('incomplete_executable_config', 'Executable compatibility profile needs source fingerprints')
            self._resolved[id] = ResolvedConfig(canonical({**preset, 'components': resolved}))

    def resolve(self, id):
        _identifier(id)
        suffix = '__unlimited_v105'
        if id.endswith(suffix):
            base = self.resolve(id[:-len(suffix)]).value
            benchmark = base['components']['benchmark']
            if benchmark['backend'] != 'robodojo':
                raise ConfigError('invalid_episode_budget_override', id)
            benchmark['episode_budget_override'] = 'robodojo_unlimited_actions_v105'
            benchmark['id'] += suffix
            _validate_component('benchmark', benchmark['id'], benchmark)
            base['id'] = id
            base['description'] += ' Operator-selected unlimited episode actions; per-call guards and native task rules retained.'
            return ResolvedConfig(canonical(base))
        try:return self._resolved[id]
        except KeyError as exc:raise ConfigError('unknown_config', id) from exc

    def catalog(self, *, env_id=None):
        values = []
        for config in self._resolved.values():
            if env_id is not None:
                try:config.bind_task(env_id=env_id, task='catalog filter', seed=0)
                except ConfigError:continue
            values.append(config.public())
        return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    listing = sub.add_parser('list'); listing.add_argument('--env-id')
    show = sub.add_parser('show'); show.add_argument('config_id')
    args = parser.parse_args()
    try:
        registry = ConfigRegistry()
        value = registry.catalog(env_id=args.env_id) if args.command == 'list' else registry.resolve(args.config_id).public()
        print(json.dumps(value, ensure_ascii=False, indent=2))
    except ConfigError as exc:
        print(json.dumps({'ok':False,'code':exc.code,'message':str(exc)}));raise SystemExit(2)


if __name__ == '__main__':main()
