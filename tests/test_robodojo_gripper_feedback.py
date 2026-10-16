from types import SimpleNamespace
import numpy as np

from sim.envs.robodojo import checked_control as module


def controller():
 c=module.CheckedController.__new__(module.CheckedController)
 c.roles=['left'];c.allowed_pairs=[];c.contact_grants={};c.gripper_intents=[0.]
 c.native=SimpleNamespace(sim=SimpleNamespace(physics_dt=.004))
 c.world=SimpleNamespace(support_pairs=[],refresh=lambda:None,positions=lambda:[np.array([0.])],
  scene=SimpleNamespace(snapshot={'robots':[{'joint_names':['j'],'arm_joint_names':['j']}]},check=lambda *a,**k:{'clear':True,'clearance_m':1.}),
  public_policy=lambda:{})
 c.adapter=SimpleNamespace(_terminal=False,_last_obs={'proprio':{'arms':{'left':{'gripper_state':{'openness':0.}}}}})
 return c


def test_full_closure_observes_sustained_stability_before_returning():
 c=controller();calls=[]
 def action(targets,grippers):
  calls.append((targets,grippers))
  for _ in range(10):c.holding_motion.record({'left':np.eye(4)});c.physics_steps+=1
 c._action=action
 r=c.gripper('left',0.)
 assert len(calls)==r['native_actions']==11
 assert r['physics_steps']==110
 assert r['holding_motion']['roles']['left']['continuous_evidence_samples']==101
 assert abs(r['holding_motion']['roles']['left']['window_sim_s']-.4)<1e-10
 assert not r['grasp_confirmed'] and not r['holding_motion']['grasp_confirmed']


def test_zero_dispatched_actions_cannot_claim_measurement():
 c=controller();c._action=lambda *a:(_ for _ in ()).throw(AssertionError('Must not dispatch'))
 r=c.gripper('left',0.,max_actions=0)
 assert r['native_actions']==r['physics_steps']==0
 assert r['holding_motion']['status']=='no_measurements'


def test_release_reports_no_current_attachment_and_expires_grant():
 c=controller();c.contact_grants={'left':{'indices':[],'pairs':[]}}
 c._rebuild_allowed_pairs=lambda:None
 def action(*a):
  c.adapter._last_obs['proprio']['arms']['left']['gripper_state']['openness']=1.
  c.holding_motion.record({});c.physics_steps+=1
 c._action=action
 r=c.gripper('left',1.)
 assert r['holding_motion']['status']=='no_current_two_finger_evidence'
 assert not r['contact_authorization_retained']


def test_guard_records_only_completed_physics_steps(monkeypatch):
 import importlib
 attachments=importlib.import_module('sim.envs.robodojo.attached_geometry')
 c=controller();c.operation='gripper';c.stopped=None;c.physics_steps=0;c.recovery=None
 holding=importlib.import_module('sim.envs.robodojo.holding_motion')
 c.holding_motion=holding.HoldingMotionWindow(.004)
 c.arm_start=[np.array([0.])];c.arm_targets=[np.array([0.])]
 c.world.scene.snapshot['robots'][0]['joint_limits']=[[-1.,1.]]
 c.world.scene.check_path=lambda *a,**k:{'status':'clear'}
 c.native.robot_manager=SimpleNamespace(robot_key=[SimpleNamespace(data=SimpleNamespace(joint_pos_target=np.array([[0.]])))])
 c._contact_scene=lambda s:s
 c._active_contacts=lambda:[]
 c._workpiece_collision_pairs=lambda:[]
 frames={'left':np.eye(4)}
 monkeypatch.setattr(attachments,'attached_scene',lambda *a:(c.world.scene,[{'role':'left'}]))
 monkeypatch.setattr(attachments,'attachment_frames',lambda world,pos,grants,evidence:frames if evidence else {})
 original_calls=[]
 c._guarded_step(lambda:original_calls.append(1))
 assert len(original_calls)==c.physics_steps==len(c.holding_motion.samples)==1
 c.stopped={'reason':'fixture_stop'}
 c._guarded_step(lambda:original_calls.append(1))
 assert len(original_calls)==len(c.holding_motion.samples)==1


def test_full_closure_respects_explicit_original_cap():
 from scipy.spatial.transform import Rotation
 c=controller();steps=[0]
 def action(*a):
  for _ in range(10):
   steps[0]+=1;t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',steps[0]*.02).as_matrix()
   c.holding_motion.record({'left':t});c.physics_steps+=1
 c._action=action;r=c.gripper('left',0.,max_actions=25)
 assert r['native_actions']==25 and r['gripper_settling']['status']=='budget_exhausted'
 assert r['host_adjustments'][0]['extra_actions_after_width_stable']==21
 assert not r['grasp_confirmed']


def test_stable_motion_completes_before_cap_and_reports_extra_physics():
 from scipy.spatial.transform import Rotation
 c=controller();steps=[0]
 def action(*a):
  for _ in range(10):
   steps[0]+=1;t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',min(steps[0],50)*.02).as_matrix()
   c.holding_motion.record({'left':t});c.physics_steps+=1
 c._action=action;r=c.gripper('left',0.)
 assert r['native_actions']==15 and r['gripper_settling']['status']=='relative_motion_small'
 assert r['host_adjustments'][0]['extra_actions_after_width_stable']==11
 assert r['physics_steps']==150 and not r['grasp_confirmed']


def test_slow_persistent_rotation_cannot_pass_the_longer_window():
 from scipy.spatial.transform import Rotation
 c=controller();steps=[0]
 def action(*a):
  for _ in range(10):
   steps[0]+=1;t=np.eye(4)
   # 0.05rad/s moves only 0.008rad in the former 0.16s window.
   t[:3,:3]=Rotation.from_euler('z',steps[0]*.004*.05).as_matrix()
   c.holding_motion.record({'left':t});c.physics_steps+=1
 c._action=action;r=c.gripper('left',0.)
 assert r['native_actions']==50
 assert r['gripper_settling']['status']=='budget_exhausted'
 assert r['holding_motion']['roles']['left']['maximum_relative_rotation_rad']>.019
 assert not r['grasp_confirmed']


def test_opening_does_not_wait_for_released_body_to_settle():
 from scipy.spatial.transform import Rotation
 c=controller();steps=[0]
 def action(*a):
  c.adapter._last_obs['proprio']['arms']['left']['gripper_state']['openness']=1.
  for _ in range(10):
   steps[0]+=1;t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',steps[0]*.02).as_matrix()
   c.holding_motion.record({'left':t});c.physics_steps+=1
 c._action=action;r=c.gripper('left',1.)
 assert r['native_actions']==4 and r['gripper_settling']['status']=='not_requested'
 assert not r['host_adjustments']


def test_delayed_settling_can_finish_beyond_former_default_cap():
 from scipy.spatial.transform import Rotation
 c=controller();steps=[0]
 def action(*a):
  for _ in range(10):
   steps[0]+=1;t=np.eye(4)
   t[:3,:3]=Rotation.from_euler('z',min(steps[0],240)*.02).as_matrix()
   c.holding_motion.record({'left':t});c.physics_steps+=1
 c._action=action;r=c.gripper('left',0.)
 assert r['native_actions']==34
 assert r['gripper_settling']['status']=='relative_motion_small'
 assert r['gripper_settling']['native_action_limit']==50
 assert r['host_adjustments'][0]['extra_actions_after_width_stable']==30
 assert not r['grasp_confirmed']


def test_partial_command_keeps_25_action_ceiling():
 c=controller();calls=[]
 def action(*a):
  calls.append(1)
  c.adapter._last_obs['proprio']['arms']['left']['gripper_state']['openness']=.3 if len(calls)%2 else .6
 c._action=action;r=c.gripper('left',.5)
 assert r['native_actions']==25
 assert r['gripper_settling']['native_action_limit']==25
 assert r['gripper_settling']['status']=='not_requested'


def test_stable_proximity_does_not_hide_closed_physical_limit():
 c=controller()
 grip=c.adapter._last_obs['proprio']['arms']['left']['gripper_state']
 grip.update(measurement_available=True,openness_basis='first_actuated_joint_physical_limits',
  normalization={'joint_name':'finger','physical_joint_limits':[0.,.044],'sign':1},
  joint_positions={'finger':0.})
 def action(*a):
  for _ in range(10):c.holding_motion.record({'left':np.eye(4)});c.physics_steps+=1
 c._action=action;r=c.gripper('left',0.)
 assert r['gripper_settling']['status']=='relative_motion_small'
 assert r['closure_diagnostics']['status']=='near_closed_limit'
 assert not r['closure_diagnostics']['grasp_confirmed']
 assert r['native_actions']==11
