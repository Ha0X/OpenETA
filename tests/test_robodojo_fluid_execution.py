from types import SimpleNamespace as S
import numpy as np
import pytest
from test_robodojo_collision_geometry import fixture
from sim.envs.robodojo.collision_geometry import CollisionScene
from sim.envs.robodojo.fluid_execution import check_particles, FluidExecution


def setup(points):
    data=fixture();data['robots'][0]['role']='left'
    scene=CollisionScene(data)
    control=S(world=S(scene=scene))
    monitor=FluidExecution.__new__(FluidExecution);monitor.control=control;monitor.checks=0
    monitor.clouds=[(np.asarray(points,dtype=float),.0025)]
    return scene,monitor


def test_container_contact_allowed_but_robot_contact_blocked():
    scene,monitor=setup([[0,0,0]])
    assert monitor.check([[-1.]])['clear']  # liquid inside world/container geometry
    result=monitor.check([[0.]])
    assert not result['clear'] and result['role']=='left'
    assert set(result)=={'clear','reason','role','part','evidence'}


def test_sweep_detects_liquid_between_clear_endpoints():
    scene,monitor=setup([[0,0,0]])
    assert monitor.check([[-1.]])['clear'] and monitor.check([[1.]])['clear']
    assert not monitor.check_segment([np.array([-1.])],[np.array([1.])])['clear']


def test_safe_segment_and_new_liquid_snapshot():
    scene,monitor=setup([[0,0,0]])
    assert monitor.check_segment([np.array([-1.])],[np.array([-.8])])['clear']
    monitor.clouds=[(np.array([[-1.,0,0]]),.0025)]
    assert not monitor.check([[-1.]])['clear']


def test_liquid_inside_closed_robot_is_detected():
    scene,monitor=setup([[-1.,0,0]])
    assert not monitor.check([[-1.]])['clear']


def test_budget_unknown_and_policy_has_no_zero_contact_guarantee():
    scene,monitor=setup([[-1.,0,0]])
    assert monitor.check([[-1.]],deadline=0)['evidence']=='unknown_not_collision'
    policy=monitor.policy()['fluid_contact']
    assert policy['future_fluid_motion_predicted'] is False
    assert policy['continuous_fluid_safety_guarantee'] is False
    assert policy['private_particle_data_exposed'] is False


@pytest.mark.parametrize('fail_on',[1,2])
def test_live_monitor_halts_before_or_after_physics(monkeypatch,fail_on):
    import sim.envs.robodojo.attached_geometry as attachments
    from test_robodojo_gripper_feedback import controller
    c=controller();c.operation='move';c.stopped=None;c.physics_steps=0;c.recovery=None
    c.attachment_baseline={};c.attachment_monitor={'roles_with_evidence':[],'checked_physics_steps':0,'relative_motion':{}}
    c.arm_start=[np.array([0.])];c.arm_targets=[np.array([0.])]
    c.world.scene.snapshot['robots'][0]['joint_limits']=[[-1.,1.]]
    c.world.scene.check_path=lambda *a,**k:{'status':'clear'}
    c.native.robot_manager=S(robot_key=[S(data=S(joint_pos_target=np.array([[0.]])))])
    c._contact_scene=lambda s:s;c._active_contacts=lambda:[];c._workpiece_collision_pairs=lambda:[]
    c._prediction_scene=lambda current:(c.world.scene,[])
    monkeypatch.setattr(attachments,'attachment_frames',lambda *args:{})
    checks=[];physical=[];reads=[]
    def check(*a,**k):
        checks.append(1);return {'clear':len(checks)!=fail_on,'reason':'fluid_robot_proximity'}
    c.fluid=S(refresh=lambda:reads.append(1),check_segment=check,check=check,feedback=lambda r:r)
    c._halt=lambda reason,detail=None:setattr(c,'stopped',{'reason':reason})
    c._guarded_step(lambda:physical.append(1))
    assert len(physical)==c.physics_steps==fail_on-1
    assert len(reads)==fail_on
    assert c.stopped['reason']==('before_physics_fluid_check' if fail_on==1 else 'after_physics_fluid_check')
    c._guarded_step(lambda:physical.append(1));assert len(physical)==fail_on-1


@pytest.mark.parametrize('state',[[[float('nan')]],[[float('inf')]],[[1,2]],[]])
def test_missing_or_invalid_robot_state_cannot_pass(state):
    from sim.envs.robodojo.native_fluid import FluidCoverageError
    scene,monitor=setup([[0,0,0]])
    with pytest.raises(FluidCoverageError):monitor.check(state)


def test_empty_or_invalid_particle_state_cannot_pass():
    from sim.envs.robodojo.native_fluid import FluidCoverageError
    scene,monitor=setup([[0,0,0]])
    for clouds in ([],[(np.array([[0,0,0]]),-1.)],[(np.array([[float('nan'),0,0]]),.001)]):
        monitor.clouds=clouds
        with pytest.raises(FluidCoverageError):monitor.check([[-1.]])


def test_liquid_rejection_distinguishes_existing_proximity_from_future_target():
    _,monitor=setup([[0,0,0]])
    start=monitor.check_segment([np.array([0.])],[np.array([1.])])
    assert not start['clear'] and start['check_phase']=='measured_start'
    assert 'no certified escape policy' in start['guidance']
    target=monitor.check_segment([np.array([-1.])],[np.array([0.])])
    assert not target['clear'] and target['check_phase']=='commanded_target'
    assert 'no certified escape policy' not in target['guidance']
    sweep=monitor.check_segment([np.array([-1.])],[np.array([1.])])
    assert not sweep['clear'] and sweep['check_phase']=='sweep_sample'
    assert 'no certified escape policy' not in sweep['guidance']
    for result in (start,target,sweep):
        assert not any(k in result for k in ('particles','points','depth','volume','object_id'))


def test_liquid_budget_failure_at_start_does_not_claim_overlap_or_no_escape():
    _,monitor=setup([[0,0,0]])
    monitor.check=lambda *a,**k:{'clear':False,'reason':'fluid_check_budget','evidence':'unknown_not_collision'}
    result=monitor.check_segment([np.array([-1.])],[np.array([1.])])
    assert result['check_phase']=='measured_start'
    assert 'already intersects' not in result['guidance']
    assert 'no certified escape policy' not in result['guidance']
