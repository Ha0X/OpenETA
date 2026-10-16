"""Distance to the shortest commanded orientation arc, not endpoint balls."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


class OrientationArc:
    def __init__(self, start, goal):
        self.start=Rotation.from_matrix(np.asarray(start)[:3,:3]).as_quat()
        self.goal=Rotation.from_matrix(np.asarray(goal)[:3,:3]).as_quat()
        dot=float(self.start @ self.goal)
        if dot<0:
            self.goal=-self.goal
            dot=-dot
        self.angle=math.acos(float(np.clip(dot,0.,1.)))
        self.tangent=(self.goal-self.start*dot)/math.sin(self.angle) if self.angle>1e-8 else None

    def distance(self, pose):
        q=Rotation.from_matrix(np.asarray(pose)[:3,:3]).as_quat()
        best=max(abs(float(q @ self.start)),abs(float(q @ self.goal)))
        if self.tangent is not None:
            a=float(q @ self.start);b=float(q @ self.tangent)
            # Quaternion signs represent the same rotation. Stationary points
            # of |a*cos(theta)+b*sin(theta)| repeat every pi.
            theta=math.atan2(b,a)%math.pi
            if theta<=self.angle:
                best=max(best,math.hypot(a,b))
        return 2*math.acos(float(np.clip(best,0.,1.)))
