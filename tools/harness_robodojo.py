"""RoboDojo wire contracts shared by deployment inspection and future Host.

This adapter does not authorize movement. Existing server receipts lacking
official outcome provenance or calibrated cameras remain explicitly unknown.
"""
from __future__ import annotations

import numpy as np

from sim.envs.robodojo.outcome import SCHEMA, UPSTREAM_REVISION
from tools.harness_config import ConfigError
from tools.harness_state import named_robot_state


def official_outcome(payload):
    info = payload.get('info', {})
    receipt = info.get('official_outcome') if isinstance(info, dict) else None
    if not isinstance(receipt, dict):
        return {'status': 'unknown', 'benchmark_success': None,
                'reason': 'legacy_receipt_has_no_official_terminal_provenance'}
    if (receipt.get('schema_version') != SCHEMA or
            receipt.get('source') != 'robodojo.eval_env.terminal_flags' or
            receipt.get('source_contract_revision') != UPSTREAM_REVISION):
        raise ConfigError('unsupported_outcome_contract', 'Verify the deployed native outcome adapter')
    for key in ('terminal', 'native_execution_ok', 'reward_registered'):
        if type(receipt.get(key)) is not bool:
            raise ConfigError('invalid_outcome_receipt', f'{key} must be a boolean')
    if type(receipt.get('action_index')) is not int or receipt['action_index'] < 0:
        raise ConfigError('invalid_outcome_receipt', 'Missing native action index')
    status = ('unknown' if not receipt['reward_registered'] else 'running' if not receipt['terminal']
              else 'success' if receipt['native_execution_ok'] else 'failure')
    expected = True if status == 'success' else False if status == 'failure' else None
    if receipt.get('status') != status or receipt.get('benchmark_success') is not expected:
        raise ConfigError('inconsistent_outcome_receipt', 'Native flags and declared benchmark result disagree')
    return {'status': status, 'benchmark_success': expected, 'terminal': receipt['terminal'],
            'source': receipt['source'], 'action_index': receipt['action_index']}


def camera_directory(observation):
    raw = observation.get('cameras', [])
    if isinstance(raw, dict):
        frames = [{**value, 'frame_id': name} for name,value in raw.items() if isinstance(value, dict)]
    elif isinstance(raw, list):
        frames = raw
    else:
        raise ConfigError('invalid_camera_directory', 'Expected named cameras or camera array')
    result = {}
    for frame in frames:
        if not isinstance(frame, dict) or not isinstance(frame.get('frame_id'), str) or not frame['frame_id']:
            raise ConfigError('invalid_camera_directory', 'Every camera needs an explicit name')
        name = frame['frame_id']
        if name in result:
            raise ConfigError('invalid_camera_directory', 'Duplicate camera name')
        k, ext = frame.get('intrinsics'), frame.get('extrinsics')
        calibrated = False
        # A raw matrix alone does not state optical axes, world origin or depth
        # convention. Require all of these before RGB-D pixel-to-world tools.
        if isinstance(k, dict) and isinstance(ext, dict):
            try:
                k_values = np.array([k[n] for n in ('fx','fy','cx','cy')], dtype=float)
                t = np.array(ext['camera_to_world'], dtype=float)
                calibrated = (np.isfinite(k_values).all() and k_values[0] > 0 and k_values[1] > 0
                    and t.shape == (4,4) and np.isfinite(t).all()
                    and np.allclose(t[3], [0,0,0,1]) and np.allclose(t[:3,:3].T@t[:3,:3], np.eye(3), atol=1e-7)
                    and np.isclose(np.linalg.det(t[:3,:3]), 1.) and ext.get('camera_frame') == 'opencv'
                    and ext.get('reference_frame') == 'world'
                    and frame.get('depth_convention', ext.get('depth_convention')) == 'optical_z_m')
            except (KeyError, ValueError, TypeError):
                calibrated = False
        result[name] = {'frame_id': name, 'pixel_to_world_available': bool(calibrated),
                        'reason': 'explicit_calibration' if calibrated else 'camera_axes_origin_or_depth_convention_unverified'}
    return result


def inspect_packet(config, payload):
    if config.value['components']['benchmark']['backend'] != 'robodojo':
        raise ConfigError('adapter_backend_mismatch', 'RoboDojo wire adapter requires a RoboDojo preset')
    observation = payload.get('observation', payload)
    if not isinstance(observation, dict):
        raise ConfigError('missing_observation', 'No observation dictionary')
    return {'config_id': config.id, 'config_sha256': config.sha256,
            'robot': named_robot_state(config, observation), 'cameras': camera_directory(observation),
            'outcome': official_outcome(payload), 'movement_authorized': False}


def main():
    import argparse
    import json
    from pathlib import Path
    from tools.harness_config import ConfigRegistry
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-id', required=True)
    p.add_argument('--packet', type=Path, required=True, help='Operator-owned simulator JSON receipt')
    args = p.parse_args()
    try:
        print(json.dumps(inspect_packet(ConfigRegistry().resolve(args.config_id), json.loads(args.packet.read_text())), indent=2))
    except ConfigError as exc:
        print(json.dumps({'ok': False, 'code': exc.code, 'message': str(exc)}))
        raise SystemExit(2)


if __name__ == '__main__':
    main()
