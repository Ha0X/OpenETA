from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.envs.robodojo.articulation_bindings import bind_articulations


def scene():
    poses=np.array([[[0,0,0,0,0,0,1],[1,0,0,0,0,0,1]]],dtype=float)
    pv=S(link_paths=[['/object/base','/object/button']],get_link_transforms=lambda:poses.copy())
    view=S(_physics_view=pv,body_names=['base','button'],is_physics_handle_valid=lambda:True)
    obj=S(prim_path='/object',_articulation_view=view)
    return S(scene_manager=S(_articulation_objects=[{'one':obj}])),poses,view


def test_live_link_binding_tracks_each_link_and_converts_quaternion():
    native,poses,view=scene();links=bind_articulations(native)
    poses[0,1,2]=.01
    poses[0,1,3:]=[0,0,2**-.5,2**-.5]
    p,q=links['/object/button'].get_world_pose()
    assert np.allclose(p,[1,0,.01]) and np.allclose(q,[2**-.5,0,0,2**-.5])
    assert np.allclose(links['/object/base'].get_world_pose()[0],[0,0,0])
    view.is_physics_handle_valid=lambda:False
    with pytest.raises(ValueError,match='invalid'):links['/object/button'].get_world_pose()


def test_link_directory_mismatch_and_invalid_transform_fail_closed():
    native,poses,view=scene();view.body_names=['button','base']
    with pytest.raises(ValueError,match='names'):bind_articulations(native)
    view.body_names=['base','button'];poses[0,1,0]=np.nan
    with pytest.raises(ValueError,match='invalid'):bind_articulations(native)


def test_root_itself_is_a_valid_link_but_neighboring_object_is_not():
    native,poses,view=scene()
    view._physics_view.link_paths=[['/object','/object/button']]
    view.body_names=['object','button']
    assert '/object' in bind_articulations(native)
    native,poses,view=scene()
    view._physics_view.link_paths=[['/foreign/base','/object/button']]
    with pytest.raises(ValueError,match='outside'):bind_articulations(native)


def test_native_body_root_may_have_sibling_links_within_authored_asset():
    native,poses,view=scene()
    obj=native.scene_manager._articulation_objects[0]['one']
    obj.prim_path='/object/base';obj.usd_prim_path=obj._prim_path='/object'
    assert set(bind_articulations(native))=={'/object/base','/object/button'}
    view._physics_view.link_paths=[['/object/base','/foreign/button']]
    with pytest.raises(ValueError,match='outside'):bind_articulations(native)


def test_native_authored_root_disagreement_cannot_expand_ownership():
    native,poses,view=scene()
    obj=native.scene_manager._articulation_objects[0]['one']
    obj.prim_path='/foreign/base';obj.usd_prim_path=obj._prim_path='/object'
    with pytest.raises(ValueError,match='outside'):bind_articulations(native)
    obj.prim_path='/object/base';obj._prim_path='/elsewhere'
    with pytest.raises(ValueError,match='outside'):bind_articulations(native)
