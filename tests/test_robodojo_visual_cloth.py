import math
import numpy as np
from test_robodojo_cloth_execution import coordinator, pose
from test_robodojo_deformable_collision import make, probe
from sim.envs.robodojo.cloth_execution import VISUAL_MODE
from sim.envs.robodojo.deformable_collision import ClothCollisionSnapshot


def test_visual_grasp_monitor_never_reads_private_geometry_or_stops_on_loss():
    c=coordinator();c.visual_only=True
    # No readers, monitors, native env or gripper state exist in this fixture.
    c.measure()
    c.prepare_move({'left':[pose(.01)]},'independent')
    c.evidence={'left':{'proximity':'both_fingers_near_patch'}}
    receipt=c.decorate({'reached_target':True},'move')
    assert 'local_contacts' not in receipt['cloth_monitor']
    assert not receipt['cloth_monitor']['grasp_confirmed']
    assert not receipt['cloth_monitor']['mesh_grasp_inference']
    assert c.policy()['deformables']==VISUAL_MODE


def test_visual_all_batches_sync_without_hidden_grasp_state():
    c=coordinator();c.visual_only=True
    targets={'left':[pose(.2),pose(.3)],'right':[pose(.1)]}
    effective,mode=c.prepare_move(targets,'independent')
    assert mode=='synchronized'
    assert np.isclose(effective['left'][0][0,3],.05)
    assert np.isclose(effective['right'][0][0,3],.025)
    assert any(e['kind']=='cloth_batch_synchronized' for e in c.events)
    assert c.decorate({'reached_target':True},'move')['continuation_required']
    assert targets['left'][0][0,3]==.2


def test_finger_permission_does_not_allow_arm_or_another_role():
    c=ClothCollisionSnapshot(make().frame,permitted_finger_roles=['left'])
    point=probe(.04,.04,0)
    assert c.clearance(point,role='left',finger=True)==math.inf
    assert c.clearance(point,role='left',finger=False)<0
    assert c.clearance(point,role='right',finger=True)<0


def test_visual_safety_check_still_rejects_arm_cloth_collision():
    from test_robodojo_collision_geometry import fixture
    from sim.envs.robodojo.collision_geometry import CollisionScene
    data=fixture();data['robots'][0]['role']='left'
    scene=CollisionScene(data);c=ClothCollisionSnapshot(make().frame,permitted_finger_roles=['left'])
    assert not c.check(scene,[[0]],allowed_world_ids=['world'])['clear']
