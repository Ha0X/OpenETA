"""Bounded relative-motion measurements, never a grasp-success classifier."""
from collections import deque
import math
import numpy as np
from scipy.spatial.transform import Rotation


class HoldingMotionWindow:
    def __init__(self, physics_dt, *, window_s=.16):
        if not math.isfinite(physics_dt) or physics_dt<=0:
            raise ValueError('A measured positive physics period is required')
        self.period=float(physics_dt)
        self.samples=deque(maxlen=min(512,max(2,math.ceil(window_s/self.period)+1)))
        self.step=0

    def record(self, frames):
        self.step+=1
        self.samples.append((self.step,{role:np.array(frame,copy=True) for role,frame in frames.items()}))

    def public(self):
        if not self.samples:
            return {'status':'no_measurements','grasp_confirmed':False,'roles':{}}
        roles={}
        for role in self.samples[-1][1]:
            trailing=[]
            for step,frames in reversed(self.samples):
                if role not in frames:break
                trailing.append((step,frames[role]))
            trailing.reverse()
            reference=trailing[0][1]
            positions=[float(np.linalg.norm(frame[:3,3]-reference[:3,3])) for _,frame in trailing]
            rotations=[float(Rotation.from_matrix(frame[:3,:3]@reference[:3,:3].T).magnitude()) for _,frame in trailing]
            roles[role]={'continuous_evidence_samples':len(trailing),
                'window_sim_s':(trailing[-1][0]-trailing[0][0])*self.period,
                'maximum_relative_translation_m':max(positions),
                'maximum_relative_rotation_rad':max(rotations),
                'evidence':'authorized_body_with_both_fingers_within_2mm',
                'reference':'earliest_continuous_sample_in_recent_window',
                'grasp_confirmed':False}
        return {'status':'measured' if roles else 'no_current_two_finger_evidence',
            'grasp_confirmed':False,'roles':roles,
            'scope':'relative_motion_during_this_gripper_call_only',
            'limitation':'Stationary proximity and small motion do not prove load-bearing support. Inspect images and use a checked lift before free-space transport.'}
