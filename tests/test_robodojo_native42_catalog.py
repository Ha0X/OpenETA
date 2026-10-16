"""Catalog presence never substitutes for native execution admission."""
import json
from pathlib import Path
from sim.envs.robodojo.layout import SUPPORT_REQUIREMENTS

ROOT=Path(__file__).resolve().parents[1]


def test_full_native_catalog_preserves_native_layout_and_missing_capabilities():
    inventory=json.loads((ROOT/'configs/harness/robodojo_rigid32_inventory.json').read_text())['tasks']
    entries=json.loads((ROOT/'sim/envs/robodojo/dual_task_manifest.json').read_text())['tasks']
    by_name={e['task']:e for e in entries}
    assert len(entries)==len(by_name)==42
    assert set(by_name)=={e['task'] for e in inventory}
    added=[e for e in entries if e.get('source')=='native42_catalog_registration']
    assert len(added)==12
    for original in inventory:
        entry=by_name[original['task']]
        if entry not in added:continue
        assert entry['dual_arm_status']=='untested'
        assert entry['required_arms']==2
        assert entry['native_robot_config']==original['native_robot_config']
        if 'Fluid' in original['scene_types']:
            assert 'live_fluid_adapter' in entry['execution_requirements']
        if original['native_robot_config']=='dual_x5_and_franka_competition':
            assert set(SUPPORT_REQUIREMENTS)<=set(entry['execution_requirements'])
