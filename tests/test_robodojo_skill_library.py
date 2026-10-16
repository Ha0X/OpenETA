import copy
import json
from pathlib import Path
import shutil

import pytest
from tools.harness_config import ConfigRegistry
from tools.robodojo_skill_library import BUNDLES, PREFIX, SkillLibrary
from test_codex_robodojo_host import host

ROOT=Path(__file__).resolve().parents[1]


def pins():
    return ConfigRegistry().resolve('robodojo_dual_x5_checked_v1').value['components']['interface']['source_sha256']


def test_fixed_catalog_reads_each_guide_with_pinned_contracts():
    library=SkillLibrary(pins())
    for skill_id, paths in BUNDLES.items():
        receipt=library.read(skill_id)
        assert [d['reference'] for d in receipt['documents']]==list(paths)
        assert receipt['native_actions']==0 and not receipt['capabilities_changed']
        for doc in receipt['documents']:
            assert doc['text']==(ROOT/PREFIX/doc['reference']).read_text()
            assert doc['sha256']==pins()[PREFIX+doc['reference']]


@pytest.mark.parametrize('skill_id',['../../secret','references/pick-place.md','align_blocks','/etc/passwd'])
def test_no_arbitrary_paths_or_task_solution_lookup(skill_id):
    with pytest.raises(ValueError,match='Unknown skill'):SkillLibrary(pins()).read(skill_id)


def test_snapshot_is_pinned_cached_and_returned_by_copy(tmp_path):
    root=tmp_path/'plugin';shutil.copytree(ROOT/PREFIX,root)
    library=SkillLibrary(pins(),root=root)
    first=library.read('push-slide')
    (root/'references/push-slide.md').write_text('changed after startup')
    assert library.read('push-slide')==first
    first['documents'][0]['text']='caller edit'
    assert library.read('push-slide')!=first
    with pytest.raises(ValueError,match='differs'):SkillLibrary(pins(),root=root)


def test_symlink_escape_rejected_even_when_bytes_match(tmp_path):
    root=tmp_path/'plugin';shutil.copytree(ROOT/PREFIX,root)
    original=root/'references/push-slide.md';external=tmp_path/'other.md'
    external.write_bytes(original.read_bytes());original.unlink();original.symlink_to(external)
    with pytest.raises(ValueError,match='escapes'):SkillLibrary(pins(),root=root)


def test_host_document_read_does_not_call_simulator_or_change_robot_state(tmp_path):
    h,remote=host(tmp_path,dual=True)
    try:
        before=copy.deepcopy(h.packet);calls=len(remote.calls);count=h.tool_calls
        result=h.call('read_skill',{'skill_id':'push-slide'})
        payload=json.loads(result[0].text)
        assert 'error' not in payload or payload['error'] is None
        assert payload['receipt']['kind']=='packaged_skill_guidance'
        assert len(remote.calls)==calls and h.packet==before and h.tool_calls==count+1
        assert all(item.type=='text' for item in result)
        evidence=json.loads((h.output/'calls/000001/response.json').read_text())
        assert evidence['content'][0]['text']==result[0].text
        bad=json.loads(h.call('read_skill',{'skill_id':'../summary.json'})[0].text)
        assert bad['error']['code']=='request_rejected' and len(remote.calls)==calls
        assert not h.closed
    finally:h.close('test')
