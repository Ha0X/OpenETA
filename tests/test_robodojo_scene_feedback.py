from types import SimpleNamespace as S
from sim.envs.robodojo.native_scene import NativeScene


def test_missing_live_articulations_does_not_claim_task_interaction_is_forbidden():
    scene=NativeScene.__new__(NativeScene)
    scene.scene=S(items=[],snapshot={'robots':[]})
    scene.support_pairs=[];scene.table_geometry_profile='absent_no_mount_exemption'
    scene.articulated={}
    policy=scene.public_policy()
    assert policy['world_articulations']=='none_admitted'
    assert 'not a prohibition' in policy['world_articulations_note']
    assert 'movable joint' in policy['world_articulations_note']
    scene.articulated={'private_link':[]}
    policy=scene.public_policy()
    assert 'Live world articulation links' in policy['world_articulations_note']
    assert 'private_link' not in str(policy)
