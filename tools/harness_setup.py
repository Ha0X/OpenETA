"""Preparation-only MCP: select a preset before starting a robot session.

The operator supplies task identity and output path. The Agent supplies only a
registered ID and its catalog hash. No simulator, authentication or motion runs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from threading import Lock
import time

from tools.harness_config import ConfigError, ConfigRegistry, _keys, _read


class Preparation:
    def __init__(self, *, env_id, task, seed, output, registry=None):
        self.registry = registry or ConfigRegistry()
        self.env_id, self.task, self.seed = env_id, task, seed
        self.output = Path(output)
        self._selection = None
        self._lock = Lock()

    def catalog(self):
        return {'phase': 'preparation', 'configs': self.registry.catalog(env_id=self.env_id),
                'selection_fixed': self._selection is not None}

    def audit(self, name, code):
        with self._lock:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            with self.output.with_suffix('.mcp.jsonl').open('a') as stream:
                stream.write(json.dumps({'time':time.time(), 'tool':name, 'code':code}) + '\n')

    def select(self, config_id, config_sha256):
        with self._lock:
            config = self.registry.resolve(config_id)
            if config.sha256 != config_sha256:
                raise ConfigError('config_hash_mismatch', 'Read the current configuration catalog before selecting')
            value = config.bind_task(env_id=self.env_id, task=self.task, seed=self.seed)
            config.require_executable()
            encoded = json.dumps(value, sort_keys=True)
            if self._selection is not None:
                if encoded != self._selection:
                    raise ConfigError('config_already_selected', 'A new preparation session is required to change the binding')
                return {'ok': True, 'binding': json.loads(self._selection), 'idempotent': True}
            self.output.parent.mkdir(parents=True, exist_ok=True)
            # One file is created exclusively; independent preparation processes
            # cannot overwrite each other's choice. Readers reject partial JSON.
            try:
                with self.output.open('x') as stream:
                    stream.write(encoded + '\n')
            except FileExistsError as exc:
                raise ConfigError('selection_output_exists', 'Use a new selection output for a new session') from exc
            self._selection = encoded
            return {'ok': True, 'binding': value, 'idempotent': False}


def read_selection(path, *, env_id, task, seed, registry=None):
    value = _read(Path(path))
    _keys(value, {'schema_version', 'config_id', 'config_sha256', 'task_instance', 'binding_sha256'}, 'selection')
    config = (registry or ConfigRegistry()).resolve(value['config_id'])
    wanted = config.bind_task(env_id=env_id, task=task, seed=seed)
    if value != wanted:
        raise ConfigError('selection_binding_mismatch', 'Selection must match the current preset and exact task instance')
    config.require_executable()
    return config


def server_for(preparation):
    from mcp.server.fastmcp import FastMCP
    server = FastMCP('openeta-setup', instructions='Select an executable configuration for the assigned task. This preparation service has no robot controls. Selection is fixed; execution uses a separate session.')

    @server.tool()
    def list_harness_configs() -> dict:
        """List compatible registered presets and their missing capabilities."""
        result = preparation.catalog()
        preparation.audit('list_harness_configs', 'ok')
        return result

    @server.tool()
    def select_harness_config(config_id: str, config_sha256: str) -> dict:
        """Bind a catalog ID/hash to the operator-assigned task. No motion occurs."""
        try:
            result = preparation.select(config_id, config_sha256)
            preparation.audit('select_harness_config', 'ok')
            return result
        except ConfigError as exc:
            preparation.audit('select_harness_config', exc.code)
            return {'ok': False, 'code': exc.code, 'message': str(exc), 'phase': 'preparation'}

    return server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env-id', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(sys.argv[1:] or json.loads(os.environ.get('OPENETA_SETUP_ARGS', '[]')))
    if args.output.exists():
        p.error('--output must be a new file')
    server_for(Preparation(**vars(args))).run(transport='stdio')


if __name__ == '__main__':
    main()
