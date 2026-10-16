"""Stop-policy tests use robot commands only, with no scene/physics dependency."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim.envs.robodojo.checked_control import CheckedController

class Tensor(np.ndarray):
    def clone(self): return self.copy()
    def new_tensor(self, value): return np.asarray(value).view(Tensor)

def tensor(value): return np.asarray(value,dtype=float).view(Tensor)

class Queue:
    def __init__(self): self.items=[1,2]
    def is_empty(self): return not self.items
    def pop(self): return self.items.pop()

class Entity:
    def __init__(self, failure=None):
        self.data=S(joint_pos=tensor([[.1,.2,.021,.026]]),joint_vel=tensor([[1.,2.,3.,4.]]))
        self.drive=tensor([[.3,.4,-.001,.003]])
        self.failure=failure;self.reads=0;self.writes=0
        self.root_physx_view=S(get_dof_position_targets=self.get_targets)
    def get_targets(self):
        self.reads+=1
        if self.failure=='raise' or (self.failure=='readback' and self.reads>1):raise RuntimeError('unavailable')
        if self.failure=='nan':return tensor([[0,0,float('nan'),0]])
        if self.failure=='shape':return tensor([[0,0]])
        return self.drive.copy()
    def set_joint_position_target(self,target):self.drive=target.copy()
    def set_joint_velocity_target(self,target):self.velocity=target.copy()
    def write_data_to_sim(self):self.writes+=1

def control(failure=None):
    e=Entity(failure);q=Queue()
    c=CheckedController.__new__(CheckedController)
    c.stopped=None;c.physics_steps=20;c.roles=['left'];c.gripper_intents=[0.]
    c.native=S(robot_manager=S(control_manager=S(control_queue=[q]),robot_key=[e],
        robot_list=[S(ee_type='gripper',gripper_joint_indices=[2,3])]))
    return c,e,q

@pytest.mark.parametrize('preserve',[False,True])
def test_stop_holds_arms_clears_queue_and_preserves_only_requested_finger_targets(preserve):
    c,e,q=control();previous=e.drive.copy()
    c._halt('requested_execution_horizon_reached',preserve_gripper_targets=preserve)
    np.testing.assert_array_equal(e.drive[:,:2],e.data.joint_pos[:,:2])
    np.testing.assert_array_equal(e.drive[:,2:],previous[:,2:] if preserve else e.data.joint_pos[:,2:])
    np.testing.assert_array_equal(e.velocity,np.zeros((1,4)))
    assert q.is_empty() and e.writes==1 and c.physics_steps==20 and c.gripper_intents==[0.]
    h=c.stopped['actuator_hold'];g=h['grippers'][0]
    assert h['extra_physics_steps']==0 and not h['grasp_or_force_confirmed']
    assert g['readback_matches_requested'] and g['previous_position_targets']==[-.001,.003]
    assert g['policy']==('previous_position_targets' if preserve else 'measured_position')

@pytest.mark.parametrize('failure',['raise','nan','shape'])
def test_unavailable_prior_target_cannot_prevent_measured_safety_hold(failure):
    c,e,q=control(failure)
    c._halt('requested_execution_horizon_reached',preserve_gripper_targets=True)
    np.testing.assert_array_equal(e.drive,e.data.joint_pos)
    assert q.is_empty() and e.writes==1
    assert c.stopped['actuator_hold']['grippers'][0]['policy']=='measured_position_fallback'

def test_readback_failure_does_not_claim_verified_target_or_change_first_stop_reason():
    c,e,q=control('readback');c.stopped={'reason':'measured_collision_after_physics'}
    c._halt('physics_monitor_exception')
    assert c.stopped['reason']=='measured_collision_after_physics'
    g=c.stopped['actuator_hold']['grippers'][0]
    assert g['readback_status']=='unavailable' and 'readback_matches_requested' not in g
    np.testing.assert_array_equal(e.drive,e.data.joint_pos)


def test_halt_never_writes_task_owned_support_drive_targets():
    c,e,q=control();support=Entity();previous=support.drive.copy()
    c.native.robot_manager.robot_key.append(support)
    c.native.robot_manager.robot_list.append(S(type='support',ee_type='gripper',gripper_joint_indices=[2,3]))
    c._halt('requested_execution_horizon_reached',preserve_gripper_targets=True)
    assert support.writes==0 and support.reads==0 and not hasattr(support,'velocity')
    np.testing.assert_array_equal(support.drive,previous)
    assert len(c.stopped['actuator_hold']['grippers'])==1 and e.writes==1


def test_safety_hold_discloses_changed_drive_targets_without_claiming_grasp():
    c,e,_=control();c._halt('physical_contact_penetration_limit')
    g=c.stopped['actuator_hold']['grippers'][0]
    assert g['drive_targets_changed'] is True
    assert 'unselected arm' in g['guidance'] and 'not guaranteed' in g['guidance']
    np.testing.assert_array_equal(e.drive,e.data.joint_pos)
    c,e,_=control();previous=e.drive.copy()
    c._halt('requested_execution_horizon_reached',preserve_gripper_targets=True)
    g=c.stopped['actuator_hold']['grippers'][0]
    assert g['drive_targets_changed'] is False and 'retained' in g['guidance']
    np.testing.assert_array_equal(e.drive[:,2:],previous[:,2:])


@pytest.mark.parametrize('preserve', [False, True])
@pytest.mark.parametrize('intent', [0., .4, 1.])
def test_stop_and_next_request_keep_intent_separate_from_readback(preserve, intent):
    c, e, _ = control()
    c.gripper_intents = [intent]
    # Native command readback and finger positions can differ from the request.
    c.adapter = S(_last_obs={'proprio': {'arms': {'left': {
        'gripper_command': .512, 'gripper_state': {'openness': .6}}}}})
    c._halt('tracking_progress_stalled' if preserve else 'measured_collision_after_physics',
            preserve_gripper_targets=preserve)
    # A subsequent motion, recovery or wait obtains commands through _grippers.
    resumed = c._grippers()
    assert resumed == [intent]
    resumed[0] = .9
    assert c._grippers() == [intent]  # A caller cannot mutate the latched intent.
    if not preserve:
        guidance = c.stopped['actuator_hold']['grippers'][0]['guidance']
        assert 'a later accepted motion or wait reapplies it' in guidance
        assert 'normal safety checks' in guidance
    # A later explicit release must win over the old grip, regardless of readback.
    c.gripper_intents[0] = 1.
    assert c._grippers() == [1.]
