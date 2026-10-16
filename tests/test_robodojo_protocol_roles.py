from adapter.protocol import RobotState


def test_single_arm_wire_remains_unchanged_when_no_roles_are_present():
    state=RobotState.from_dict({'joint_positions':[.1,.2],'ee_pose':[1,2,3,0,0,0,1],'gripper_open':1.})
    assert state.to_dict()=={'joint_positions':[.1,.2],'joint_velocities':[],
        'end_effector_pose':{'xyz':[1.,2.,3.],'quat_xyzw':[0.,0.,0.,1.]},
        'gripper_state':{'openness':1.,'open':True}}


def test_dual_arm_wire_preserves_independent_joint_names_poses_and_calibration():
    state={'arms':{role:{'joint_positions':[float(i)]*6,'joint_names':[f'joint{j}' for j in range(1,7)],
                        'ee_pose':[i,0,1,0,0,0,1], 'gripper_state':{'openness':float(i)},
                        'metadata':{'reported_link':'link6','role_binding':role}}
                   for i,role in enumerate(['left','right'])}}
    wire=RobotState.from_dict(state).to_dict()
    assert set(wire['arms'])=={'left','right'}
    assert wire['arms']['left']['end_effector_pose']['xyz']==[0.,0.,1.]
    assert wire['arms']['right']['end_effector_pose']['xyz']==[1.,0.,1.]
    assert wire['arms']['left']['gripper_state']['openness']==0.
    assert wire['arms']['right']['gripper_state']['openness']==1.
    assert wire['arms']['right']['joint_names']==[f'joint{j}' for j in range(1,7)]
    assert wire['arms']['right']['metadata']['role_binding']=='right'
