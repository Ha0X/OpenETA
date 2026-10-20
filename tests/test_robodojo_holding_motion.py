import numpy as np
from scipy.spatial.transform import Rotation

from sim.envs.robodojo import holding_motion as m

def pose(x=0.,angle=0.):
 t=np.eye(4);t[0,3]=x;t[:3,:3]=Rotation.from_euler('z',angle).as_matrix();return t

def test_excursion_is_not_hidden_by_returning_to_start():
 w=m.HoldingMotionWindow(.004)
 for t in [pose(),pose(.001,.03),pose()]:w.record({'left':t})
 r=w.public()['roles']['left']
 assert r['maximum_relative_translation_m']==.001
 assert abs(r['maximum_relative_rotation_rad']-.03)<1e-10
 assert r['window_sim_s']==.008 and not r['grasp_confirmed']

def test_missing_evidence_breaks_window_not_cached_grasp():
 w=m.HoldingMotionWindow(.004)
 w.record({'left':pose()});w.record({})
 assert w.public()['status']=='no_current_two_finger_evidence'
 w.record({'left':pose(.1)})
 assert w.public()['roles']['left']['continuous_evidence_samples']==1
 assert w.public()['roles']['left']['window_sim_s']==0

def test_window_bounded_and_reports_each_arm_independently():
 w=m.HoldingMotionWindow(.004)
 for i in range(1000):w.record({'left':pose(i*.001),'right':pose()})
 assert len(w.samples)==41
 r=w.public()['roles']
 assert abs(r['left']['window_sim_s']-.16)<1e-10
 assert abs(r['left']['maximum_relative_translation_m']-.04)<1e-10
 assert r['right']['maximum_relative_translation_m']==0
 assert w.public()['grasp_confirmed'] is False
