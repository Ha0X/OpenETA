from argparse import Namespace
import json
import subprocess

import pytest

from test_codex_plugin_launcher import fake_login
from scripts import codex_harness_prepare as launcher


def args(tmp_path):
    return Namespace(env_id='openeta/libero_libero_object_task0-v0', task='Pick up the alphabet soup', seed=0,
        output=tmp_path/'preparation', model='gpt-6-astra', effort='high', model_catalog_json=None,
        timeout=60, codex='codex')


def test_preparation_plugin_has_only_setup_and_private_auth(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source/'auth.json').read_bytes()
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    monkeypatch.setenv('OPENETA_LLM_PROVIDER', 'test-only')
    installs = []
    monkeypatch.setattr(subprocess, 'run', lambda command, **kw: installs.append(command))
    options = args(tmp_path)
    home,work,env = launcher.prepare(options)
    assert (source/'auth.json').read_bytes() == before
    assert 'OPENAI_API_KEY' not in env and 'OPENETA_LLM_PROVIDER' not in env
    assert home.stat().st_mode & 0o777 == 0o700
    assert (home/'auth.json').stat().st_mode & 0o777 == 0o600
    assert not (options.output/'marketplace/plugins/openeta').exists()
    config = (home/'config.toml').read_text()
    assert 'openeta-setup@openeta-codex-smoke' in config and 'shell_tool = false' in config
    mcp = json.loads((options.output/'marketplace/plugins/openeta-setup/.mcp.json').read_text())
    assert set(mcp) == {'openeta-setup'}
    assert mcp['openeta-setup']['args'] == ['-m','tools.harness_setup']
    assert len(installs) == 2


def test_install_failure_cleans_only_private_credentials(tmp_path, monkeypatch):
    source = fake_login(tmp_path, monkeypatch)
    before = (source/'auth.json').read_bytes()
    def fail(*a, **kw):
        raise subprocess.CalledProcessError(1,['codex','plugin','add'])
    monkeypatch.setattr(subprocess,'run',fail)
    options = args(tmp_path)
    assert launcher.run(options) == 1
    assert (source/'auth.json').read_bytes() == before
    assert not (options.output/'codex-home/auth.json').exists()
    assert not json.loads((options.output/'summary.json').read_text())['integration_passed']


def test_model_exit_zero_without_selection_is_not_success(tmp_path, monkeypatch):
    fake_login(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw: None)
    class Process:
        pid = 99999999
        def poll(self):return 0
        def wait(self, **kw):return 0
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**kw: Process())
    monkeypatch.setattr(launcher.ownership,'cleanup',lambda *a: [])
    options = args(tmp_path)
    assert launcher.run(options) == 1
    summary = json.loads((options.output/'summary.json').read_text())
    assert not summary['integration_passed'] and summary['private_auth_removed']


def test_existing_preparation_cannot_be_overwritten(tmp_path):
    options = args(tmp_path)
    options.output.mkdir()
    marker = options.output/'selection.json'
    marker.write_text('original binding')
    with pytest.raises(ValueError, match='new directory'):
        launcher.run(options)
    assert marker.read_text() == 'original binding'
