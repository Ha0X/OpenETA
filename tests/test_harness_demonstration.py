import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from tools.harness_demonstration import load_demonstration
from scripts import codex_plugin_smoke as launcher


def bundle(tmp_path):
    image = tmp_path/'public.png';Image.new('RGB', (10,10)).save(image)
    value = {'schema': 'openeta.public_demonstration.v1', 'kind': 'task_specific',
             'public_only_reviewed': True,
             'source': {'reference': 'public://fixture', 'episode_id': 'previous', 'env_id': 'fixture', 'seed': 1},
             'text': 'A public historical observation.',
             'images': [{'path': 'public.png', 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
                         'caption': 'Historical image, not live state.'}]}
    path = tmp_path/'demo.json';path.write_text(json.dumps(value))
    return path, value


def load(path, tmp_path):
    return load_demonstration(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                              env_id='fixture', seed=0, output=tmp_path/'run')


def test_snapshot_has_exact_bytes_and_disjoint_conditioning_label(tmp_path):
    path, value = bundle(tmp_path)
    prompt, record, images = load(path, tmp_path)
    assert record['evaluation_protocol'] == 'one_shot_task_specific'
    assert record['independent_zero_shot_pass_at_k'] is False
    assert (tmp_path/'run/demonstration/manifest.json').read_bytes() == path.read_bytes()
    assert Path(images[0]).read_bytes() == (tmp_path/'public.png').read_bytes()
    assert value['text'] in prompt and value['images'][0]['caption'] in prompt


@pytest.mark.parametrize('damage', ['private', 'same_layout', 'same_task_generic', 'other_task_specific', 'path', 'symlink', 'image_hash', 'text_size', 'unknown_field'])
def test_invalid_demo_rejected_before_snapshot(tmp_path, damage):
    path, value = bundle(tmp_path)
    if damage == 'private':value['public_only_reviewed'] = False
    if damage == 'same_layout':value['source']['seed'] = 0
    if damage == 'same_task_generic':value['kind'] = 'task_independent'
    if damage == 'other_task_specific':value['source']['env_id'] = 'other'
    if damage == 'path':value['images'][0]['path'] = '../outside.png'
    if damage == 'symlink':
        external = tmp_path.parent/(tmp_path.name+'-outside.png');external.write_bytes((tmp_path/'public.png').read_bytes())
        link = tmp_path/'link.png';link.symlink_to(external);value['images'][0]['path'] = 'link.png'
    if damage == 'image_hash':value['images'][0]['sha256'] = '0'*64
    if damage == 'text_size':value['text'] = 'a'*24001
    if damage == 'unknown_field':value['private_mesh'] = []
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):load(path, tmp_path)
    assert not (tmp_path/'run/demonstration').exists()


def test_unpinned_demo_rejected(tmp_path):
    path, _ = bundle(tmp_path)
    with pytest.raises(ValueError, match='SHA256'):
        load_demonstration(path, None, env_id='fixture', seed=0, output=tmp_path/'run')


def test_launcher_passes_only_selected_images_and_labels_actual_budgets(tmp_path, monkeypatch):
    path, _ = bundle(tmp_path);commands = []
    def prepare(output, *args, **kw):
        home = output/'codex-home';home.mkdir()
        (output/'host').mkdir()
        (output/'host/host-status.json').write_text(json.dumps({'closed':True, 'cleanup':{'ok':True},
            'tool_calls':1, 'official_task_success':False, 'simulation_budget':{'unit':'native_adapter_step_calls','action_limit':200}}))
        return home, output, {}
    monkeypatch.setattr(launcher, 'prepare', prepare)
    monkeypatch.setattr(launcher.os, 'killpg', lambda *a: None)
    def popen(command, **kw):
        commands.append(command)
        return SimpleNamespace(pid=99999999, returncode=0, wait=lambda **kw:0)
    monkeypatch.setattr(launcher.subprocess, 'Popen', popen)
    args = SimpleNamespace(output=tmp_path/'run', sim_url='http://unused.invalid', env_id='fixture',
        task='fixture',seed=0,timeout=60,max_requests=400,sam3_url='',anygrasp_url='',model='fixture',effort='high',
        codex='codex',observe_only=True,controller='mink_joint_velocity',demonstration=path,
        demonstration_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert launcher.run(args) == 0
    command = commands[0];assert command[-2] == '--' and '--image' in command
    assert Path(command[command.index('--image')+1]).is_file()
    summary = json.loads((args.output/'summary.json').read_text())
    factors = summary['experiment_factors']
    assert factors['conditioning'] == 'one_shot_task_specific'
    assert factors['max_requests'] == 400 and factors['simulation_budget_observed']['action_limit'] == 200
    assert factors['model_decision_limit'] is None


def test_publisher_held_out_statement_preserves_unknown_source_seed(tmp_path):
    path, value = bundle(tmp_path)
    value['source'].update(seed=None,layout_split={'kind':'publisher_held_out_statement',
        'excluded_harness_seeds':[0,1,2,3,4], 'reference':'https://example.invalid/public-split'})
    path.write_text(json.dumps(value))
    _, record, _ = load(path,tmp_path)
    assert record['source']['seed'] is None
    assert record['layout_separation']=='publisher_attested_not_independently_verified'


@pytest.mark.parametrize('excluded', [[],[1],['0'],[False],[-1]])
def test_unknown_demo_layout_requires_applicable_holdout(tmp_path,excluded):
    path,value=bundle(tmp_path)
    value['source'].update(seed=None,layout_split={'kind':'publisher_held_out_statement',
        'excluded_harness_seeds':excluded,'reference':'https://example.invalid/public-split'})
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='held-out'):load(path,tmp_path)


def test_rgb_jpeg_demo_is_supported(tmp_path):
    path,value=bundle(tmp_path)
    image=tmp_path/'public.jpg';Image.new('RGB',(20,20)).save(image)
    value['images'][0].update(path='public.jpg',sha256=hashlib.sha256(image.read_bytes()).hexdigest())
    path.write_text(json.dumps(value))
    _,_,images=load(path,tmp_path)
    assert images[0].endswith('.jpg')
