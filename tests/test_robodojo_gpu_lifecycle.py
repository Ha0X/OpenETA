from types import SimpleNamespace as S
import pytest
from sim.envs.robodojo.gpu_lifecycle import install_scene_retirement_order


def native_class():
    class Scene:
        def __init__(self, device='cuda'):
            self.device=S(type=device);self.valid=True;self.events=[]
            self._articulation_objects=[{'hinge':'articulation'}]
            self.rigids={'box':'rigid'}
        def delete_scene_object(self, obj):
            if obj=='articulation' and self.device.type=='cuda' and not self.valid:
                raise RuntimeError('view invalidated before live joint read')
            self.events.append(obj)
            if obj=='rigid':self.valid=False
        def clear_scene_objects(self, env_id, exclude_types=None):
            for name,items in [('rigid',self.rigids),('articulation',self._articulation_objects[env_id])]:
                if name in (exclude_types or []):continue
                for key,obj in list(items.items()):
                    self.delete_scene_object(obj);del items[key]
            return 'native_result'
    return Scene


def test_native_failure_reproduced_and_gpu_order_fixed():
    cls=native_class()
    with pytest.raises(RuntimeError):cls().clear_scene_objects(0)
    install_scene_retirement_order(cls);s=cls()
    assert s.clear_scene_objects(0)=='native_result'
    assert s.events==['articulation','rigid'] and not s.rigids and not s._articulation_objects[0]
    install_scene_retirement_order(cls);s.clear_scene_objects(0)
    assert s.events==['articulation','rigid']


def test_cpu_keeps_native_order():
    cls=native_class();install_scene_retirement_order(cls);s=cls('cpu');s.clear_scene_objects(0)
    assert s.events==['rigid','articulation']


@pytest.mark.parametrize('excluded',[['articulation'],['rigid'],['articulation','rigid']])
def test_exclusion_is_not_overridden(excluded):
    cls=native_class();install_scene_retirement_order(cls);s=cls();s.clear_scene_objects(0,excluded)
    assert all(x not in s.events for x in excluded)
    assert bool(s._articulation_objects[0])==('articulation' in excluded)
    assert bool(s.rigids)==('rigid' in excluded)


def test_gpu_same_layout_retains_object_identity_but_not_fixtures():
    from sim.envs.robodojo.gpu_lifecycle import install_preallocated_layout_identity
    class Layout:
        def __init__(self,device='cuda'):
            self.scene_manager=S(device=S(type=device));self.saved_layouts=[{'seed':0}];self.counter=0
        def _generate_object_paths(self,env_idx,cat_name,model_idx,type):
            self.counter+=1
            return (f'/world/{self.counter}',f'{cat_name}_{self.counter}')
        def load_saved_layout(self,env_idx):
            return [self._generate_object_paths(env_idx,'laptop',0,'articulation'),
                    self._generate_object_paths(env_idx,'laptop',0,'articulation'),
                    self._generate_object_paths(env_idx,'box',0,'rigid'),
                    self._generate_object_paths(env_idx,'lamp',0,'light')]
    install_preallocated_layout_identity(Layout);install_preallocated_layout_identity(Layout)
    obj=Layout();first=obj.load_saved_layout(0);second=obj.load_saved_layout(0)
    assert first[:3]==second[:3] and first[0]!=first[1] and first[3]!=second[3]
    obj.saved_layouts[0]['seed']=1
    before=obj.counter
    with pytest.raises(ValueError,match='different layout'):obj.load_saved_layout(0)
    assert obj.counter==before
    cpu=Layout('cpu');first=cpu.load_saved_layout(0);second=cpu.load_saved_layout(0)
    assert first[:2]==second[:2] and first[2:]!=second[2:]
    cpu.saved_layouts[0]['seed']=1
    before=cpu.counter
    with pytest.raises(ValueError,match='different layout'):cpu.load_saved_layout(0)
    assert cpu.counter==before


def test_cpu_rigid_only_keeps_native_respawn_and_seed_changes():
    from sim.envs.robodojo.gpu_lifecycle import install_preallocated_layout_identity
    class Layout:
        def __init__(self):
            self.scene_manager=S(device='cpu');self.saved_layouts=[{'seed':0}];self.counter=0
        def _generate_object_paths(self,env_idx,cat_name,model_idx,type):
            self.counter+=1
            return (f'/world/{self.counter}', f'{cat_name}_{self.counter}')
        def load_saved_layout(self,env_idx):
            return self._generate_object_paths(env_idx,'box',0,'rigid')
    install_preallocated_layout_identity(Layout)
    obj=Layout();first=obj.load_saved_layout(0);second=obj.load_saved_layout(0)
    obj.saved_layouts[0]['seed']=1;third=obj.load_saved_layout(0)
    assert len({first,second,third})==3
