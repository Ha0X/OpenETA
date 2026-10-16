import json
import pytest
from test_codex_robodojo_host import host


@pytest.mark.parametrize('friction',[False,True])
def test_visual_cloth_has_generic_actions_and_matching_skill(tmp_path,friction):
    h,r=host(tmp_path,dual=True,visual_cloth=True,friction=friction)
    try:
        assert 'wait' in h.schemas
        schema=h.schemas['move_to'].inputSchema
        single=schema['oneOf'][0]
        assert 'contact_mark' not in single['properties'] and 'cloth_contact' not in single['properties']
        assert 'delta_xyz' in single['properties']['target']['properties']
        out=json.loads(h.call('read_skill',{'skill_id':'cloth-contact'})[0].text)
        assert out.get('physics_profile',{}).get('profile')==('cloth_friction_2x_v1' if friction else None)
        docs=out['receipt']['documents']
        assert docs[-1]['reference']=='references/contracts/cloth-visual.md'
        assert all(d['reference']!='references/contracts/cloth.md' for d in docs)
        h.call('move_to',{'role':'left','target':{'delta_xyz':[0,0,.001],'frame':'world'},'position_tolerance_m':.0005})
        request=r.calls[-1][1]['request']
        assert request['deformable_mode']=='cloth_visual_short_horizon_v2'
        assert request['contact_policy']=='physical_contact_v1'
        assert request.get('physics_profile')==('cloth_friction_2x_v1' if friction else None)
        assert request['position_tolerance_m']==.0005
        count=len(r.calls)
        bad=json.loads(h.call('gripper_control',{'role':'left','openness':0,'contact_mark':'legacy'})[0].text)
        assert bad['error']['code']=='request_rejected' and len(r.calls)==count
    finally:h.close('test')


def test_native_profile_rejects_modified_live_material(tmp_path):
    h,r=host(tmp_path,dual=True,visual_cloth=True)
    try:
        h.packet['control']['physics_profile']={'profile':'cloth_friction_2x_v1'}
        with pytest.raises(ValueError,match='physics differs'):h._validate_binding()
    finally:h.close('test')
