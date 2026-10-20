from itertools import product
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from sim.envs.robodojo.native_scene import NativeScene


def scene(suffix='8'):
    n=NativeScene.__new__(NativeScene)
    table={'id':f'/World/envs/env_0/Table/material_0122/material_0122_0_{suffix}',
           'robot_index':None,'body':None,'approximation':'convexHull',
           'vertices':list(product([-.7,.7],[-.6,.5],[.715,.765]))}
    base=np.eye(4);base[:3,3]=[-.3,-.45,.765]
    n.scene=SimpleNamespace(items=[table,{'id':'robot/base','robot_index':0,'body':'base'}],
        snapshot={'robots':[{'base_link':'base','body_frames':{'base':base.tolist()}}]})
    n.world={}
    n.env=SimpleNamespace(scene_manager=SimpleNamespace(config={'env0': {'Table': {
        'scale':[1.4,1.1,.05], 'default_pos':[0.,-.05,.74],
        'default_ori':[1.,0.,0.,0.]}}}))
    return n


@pytest.mark.parametrize('suffix',['8','12','99'])
def test_material_mesh_number_does_not_change_verified_mount(suffix):
    n=scene(suffix)
    assert n._mount_support()==[('robot/base',n.scene.items[0]['id'])]
    assert n.table_id==n.scene.items[0]['id']


@pytest.mark.parametrize('change',['height','duplicate_corner','moving','foreign_root','ambiguous','wrong_hull'])
def test_shape_or_ownership_changes_cannot_acquire_mount_exception(change):
    n=scene();t=n.scene.items[0]
    if change=='height':t['vertices']=[(x,y,z+.01) for x,y,z in t['vertices']]
    if change=='duplicate_corner':t['vertices'][0]=t['vertices'][1]
    if change=='moving':n.world[0]=object()
    if change=='foreign_root':t['id']=t['id'].replace('/Table/','/Unrelated/')
    if change=='ambiguous':n.scene.items.append(copy.deepcopy(t))
    if change=='wrong_hull':t['approximation']='none'
    with pytest.raises(ValueError,match='tabletop'):
        n._mount_support()


def test_official_narrow_offset_table_requires_both_layout_and_geometry():
    n=scene()
    n.env.scene_manager.config['env0']['Table'].update(scale=[1.,1.1,.05],default_pos=[.09,-.05,.74])
    with pytest.raises(ValueError,match='tabletop'):
        n._mount_support()  # Declared variant cannot authorize a different actual hull.
    n.scene.items[0]['vertices']=list(product([-.41,.59],[-.6,.5],[.715,.765]))
    assert n._mount_support()==[('robot/base',n.scene.items[0]['id'])]
    assert n.table_geometry_profile=='narrow_offset_v1'


@pytest.mark.parametrize('change',['missing','size','position','rotation','nonfinite'])
def test_unknown_native_table_layout_cannot_expand_support_permission(change):
    n=scene();c=n.env.scene_manager.config['env0']['Table']
    if change=='missing':c.clear()
    if change=='size':c['scale'][0]=1.5
    if change=='position':c['default_pos'][0]=.02
    if change=='rotation':c['default_ori']=[0,0,0,1]
    if change=='nonfinite':c['scale'][0]=float('nan')
    with pytest.raises(ValueError,match='tabletop'):
        n._mount_support()


def test_native_scene_without_table_has_no_mount_exemption():
    n=scene();n.env.scene_manager.config['env0'].pop('Table')
    n.env.scene_manager._tables=[None]
    with pytest.raises(ValueError,match='tabletop'):n._mount_support()
    n.scene.items=n.scene.items[1:]
    assert n._mount_support()==[]
    assert n.table_id is None and n.table_geometry_profile=='absent_no_mount_exemption'
    n.env.scene_manager._tables=[object()]
    with pytest.raises(ValueError,match='tabletop'):n._mount_support()


def test_auxiliary_base_does_not_inherit_x5_mount_exemption():
    n=scene(); base=np.eye(4);base[:3,3]=[0.,.6,.765]
    n.scene.snapshot['robots'].append({'base_link':'panda_link0','body_frames':{'panda_link0':base.tolist()}})
    n.scene.items.append({'id':'support/base','robot_index':1,'body':'panda_link0'})
    n.env.robot_manager=SimpleNamespace(robot_list=[SimpleNamespace(type='target'),SimpleNamespace(type='support')])
    assert n._mount_support()==[('robot/base',n.scene.items[0]['id'])]
