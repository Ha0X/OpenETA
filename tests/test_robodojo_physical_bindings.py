from types import SimpleNamespace
import pytest
from sim.envs.robodojo.physical_bindings import retain_rigid_bindings


def test_surviving_body_retains_live_reader_after_directory_replacement():
    state={'old_pose':1}
    old=SimpleNamespace(prim_path='/old',get_world_pose=lambda:state['old_pose'])
    new=SimpleNamespace(prim_path='/new',get_world_pose=lambda:2)
    manager=SimpleNamespace(_rigid_and_dynamic_objects=[{'active':old}])
    env=SimpleNamespace(scene_manager=manager)
    retain_rigid_bindings(env)
    manager._rigid_and_dynamic_objects[0]={'active':new}
    bindings=retain_rigid_bindings(env)
    state['old_pose']=3
    assert set(bindings)=={'/old','/new'}
    assert bindings['/old'].get_world_pose()==3
    assert bindings['/new'].get_world_pose()==2
    assert manager._rigid_and_dynamic_objects[0]=={'active':new}


def test_reused_root_refreshes_reader_and_missing_reader_is_rejected():
    old=SimpleNamespace(prim_path='/body',get_world_pose=lambda:1)
    new=SimpleNamespace(prim_path='/body',get_world_pose=lambda:2)
    env=SimpleNamespace(scene_manager=SimpleNamespace(_rigid_and_dynamic_objects=[{'body':old}]))
    retain_rigid_bindings(env)
    env.scene_manager._rigid_and_dynamic_objects[0]={'body':new}
    assert retain_rigid_bindings(env)['/body'] is new
    env.scene_manager._rigid_and_dynamic_objects[0]={'bad':SimpleNamespace(prim_path='/bad')}
    with pytest.raises(ValueError,match='live pose'):retain_rigid_bindings(env)


def test_asset_body_uses_exact_fresh_native_pose_and_rejects_invalid_handles():
    import numpy as np
    from sim.envs.robodojo.physical_bindings import AssetRigidBody
    values=np.array([[1.,2.,3.,0.,0.,0.,1.]])
    owner=SimpleNamespace(is_valid=True)
    view=SimpleNamespace(count=1,prim_paths=['/asset/belt'],get_transforms=lambda:values)
    binding=AssetRigidBody(owner,view,'/asset/belt')
    p,q=binding.get_world_pose()
    assert np.array_equal(p,[1,2,3]) and np.array_equal(q,[1,0,0,0])
    values[0,0]=7
    assert binding.get_world_pose()[0][0]==7
    assert p[0]==1
    with pytest.raises(ValueError,match='exact USD body'):
        AssetRigidBody(owner,view,'/asset/wrapper')
    values[0,6]=0
    with pytest.raises(ValueError,match='quaternion'):binding.get_world_pose()
    owner.is_valid=False
    with pytest.raises(ValueError,match='handle'):binding.get_world_pose()
