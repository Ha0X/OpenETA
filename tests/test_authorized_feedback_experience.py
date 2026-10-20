import copy
import hashlib
import json
from pathlib import Path

import pytest
from scripts.codex_plugin_smoke import authorized_feedback_experience

SOURCE=Path('evaluations/robodojo/experiments/support-v100-20261008/play-tic-tac-toe-attempt1')
ENV='openeta/robodojo_dual_x5_play_tic_tac_toe-v0'


def fixture(tmp_path):
    for name in ['authorized-experience.json','summary.json']:
        (tmp_path/name).write_bytes((SOURCE/name).read_bytes())
    return tmp_path/'authorized-experience.json'


def test_only_approved_text_reaches_prompt_and_is_not_an_independent_attempt(tmp_path):
    p=fixture(tmp_path);raw=p.read_bytes()
    prompt,record=authorized_feedback_experience(p,hashlib.sha256(raw).hexdigest(),ENV)
    assert 'turn-taking rule was violated' in prompt
    assert 'not a quotation of the model' in prompt
    assert record['independent_pass_at_k'] is False
    assert record['evaluation_protocol']=='cross_attempt_authorized_feedback'
    assert record['entry_count']==1 and record['total_characters']<=6000
    assert 'source_action_index' not in prompt and 'authorization' not in prompt


@pytest.mark.parametrize('change',['hash','env','text','outcome','action_index','cleanup'])
def test_bad_evidence_or_wrong_task_is_rejected(tmp_path,change):
    p=fixture(tmp_path);item=json.loads(p.read_text());s=json.loads((tmp_path/'summary.json').read_text());env=ENV
    if change=='env':env='openeta/robodojo_dual_x5_make_kong-v0'
    if change=='text':item['entries'].append('PRIVATE extra information')
    if change=='outcome':s['host']['outcome']['status']='running'
    if change=='action_index':item['source_action_index']+=1
    if change=='cleanup':s['host']['cleanup']['ok']=False
    p.write_text(json.dumps(item));(tmp_path/'summary.json').write_text(json.dumps(s));digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if change=='hash':digest='bad'
    with pytest.raises(ValueError):authorized_feedback_experience(p,digest,env)
