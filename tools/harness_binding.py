"""Bind a trusted preset before exposing the unchanged execution Host."""
from __future__ import annotations

from copy import copy
import json

from tools.harness_config import ConfigError, ConfigRegistry


def preflight(args, *, registry=None):
    config = (registry or ConfigRegistry()).resolve(args.harness_config)
    expected = getattr(args, 'harness_config_sha256', None)
    if expected is not None and expected != config.sha256:
        raise ConfigError('config_hash_mismatch', 'The selected preset changed; select it again before starting a new episode')
    binding = config.bind_task(env_id=args.env_id, task=args.task, seed=args.seed)
    config.require_executable()
    parts = config.value['components']
    for name, wanted in [('tool_profile', parts['interface']['tool_profile']),
                         ('expected_controller', parts['execution']['controller_option'])]:
        if getattr(args, name, None) != wanted:
            raise ConfigError('config_argument_conflict', f'{name} must be {wanted!r} for {config.id}')
    return config, binding


def validate_deployment(config, host, binding):
    """Check advertised deployment evidence, without claiming physical calibration."""
    parts = config.value['components']
    created = host.runner.environment.create_result
    if not isinstance(created, dict):
        raise ConfigError('deployment_metadata_missing', 'create_env did not return deployment metadata')
    for key, expected in [('backend', parts['benchmark']['backend']),
                          ('env_id', binding['task_instance']['env_id'])]:
        if created.get(key) != expected:
            raise ConfigError('deployment_mismatch', f'{key}: expected {expected!r}, received {created.get(key)!r}')
    spec = created.get('control_spec')
    controller = spec.get('controller') if isinstance(spec, dict) else None
    if not isinstance(controller, dict):
        raise ConfigError('deployment_metadata_missing', 'create_env.control_spec.controller is missing')
    if controller.get('controller_id') != parts['execution']['controller_id']:
        raise ConfigError('deployment_controller_mismatch', 'Live controller differs from the selected execution profile')
    for key, expected in parts['execution']['required_capabilities'].items():
        actual = controller.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise ConfigError('deployment_capability_mismatch', f'{key}: required {expected!r}, received {actual!r}')
    capabilities = host.runtime.memory.controller_capabilities() or {}
    if capabilities.get('controller_id') != controller['controller_id']:
        raise ConfigError('deployment_controller_mismatch', 'Host and simulator controller metadata disagree')
    if capabilities.get('motion_owns_trajectory_world_collision') is not True:
        raise ConfigError('deployment_collision_coverage_missing', 'LIBERO preset requires the existing per-step trajectory collision callback')
    # Current LIBERO create_env echoes an optional robot argument, often null;
    # it is not a measured joint/TCP identity. Do not manufacture that evidence.
    robot = created.get('robot')
    if robot not in (None, '') and str(robot).lower() != 'panda':
        raise ConfigError('deployment_embodiment_mismatch', f'Unexpected robot declaration: {robot!r}')
    return {'backend': created['backend'], 'env_id': created['env_id'],
            'robot_declaration': robot, 'control_spec': spec,
            'validation_scope': 'advertised_backend_controller_and_collision_coverage',
            'physical_identity_attestation': 'not_available_in_legacy_protocol',
            'calibration_basis': 'pinned_libero_implementation'}


def build_configured_host(args, *, legacy_builder):
    config, binding = preflight(args)
    if config.value['components']['interface']['factory'] == 'configured_robodojo_atomic':
        from tools.codex_robodojo_host import RoboDojoHost
        return RoboDojoHost(args, config, binding)
    # The original builder owns startup failure cleanup. Once it returns,
    # this layer owns cleanup for any binding/deployment failure.
    host = legacy_builder(copy(args))
    try:
        deployment = validate_deployment(config, host, binding)
        record = {**binding, 'resolved_config': config.value, 'deployment': deployment,
                  'status': 'bound', 'model_interface_changed': False}
        path = host.output / 'harness-binding.json'
        with path.open('x') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        # Immutable serialization avoids a mutable runtime policy dictionary.
        host.harness_binding = json.dumps(binding, sort_keys=True)
        return host
    except BaseException:
        host.close(reason='harness_binding_failed')
        raise
