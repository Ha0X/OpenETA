"""Reuse full-scene joint planning without changing agent endpoints/waypoints."""
import math
import numpy as np
from .coordination import JointTrack
from .motion_planning import plan_request


def replan_tracks(control, targets, period, timeout_s):
    trajectory, detail = plan_request(control, targets, min(15.,max(0.,timeout_s)))
    if trajectory is None:
        return None, control._public_plan(detail)
    times=[0.]
    for a,b in zip(trajectory,trajectory[1:]):
        maximum=max(float(np.max(abs(np.asarray(y)-x))) for x,y in zip(a,b))
        duration=max(period,math.ceil(maximum/.5/period)*period)
        times.append(round(times[-1]+duration,12))
    tracks=[JointTrack(role,i,times,[q[i] for q in trajectory]) for i,role in enumerate(control.roles)]
    return tracks, {'planner':'bounded_combined_joint_search',
                    'preserves':'every_agent_endpoint_and_waypoint',
                    'path':control._public_plan(detail)}
