"""Read-only diagnostics around the native stability test; verdict unchanged."""
import functools
import json
import logging
import numpy as np


def install(native):
    layout=native.scene_manager.layout_manager
    original=layout.check_layout_stability
    if getattr(original,'_openeta_diagnostics',False):return

    def array(value):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        return np.asarray(value,dtype=float).reshape(-1).copy()

    @functools.wraps(original)
    def checked(*args,**kwargs):
        records={}
        get_pose=layout.get_instance_pose
        @functools.wraps(get_pose)
        def measured(*pose_args,**pose_kwargs):
            value=get_pose(*pose_args,**pose_kwargs)
            env_idx=pose_kwargs.get('env_idx',0)
            actor=pose_kwargs.get('inst_name','unknown')
            pos,quat=map(array,value)
            key=f'{env_idx}:{actor}'
            if key not in records:
                records[key]={'first_position':pos.tolist(),'first_quaternion':quat.tolist(),
                              'max_axis_displacement_m':0.,'max_rotation_deg':0.,'min_z_m':float(pos[2]),'samples':0}
            row=records[key];row['samples']+=1
            row['max_axis_displacement_m']=max(row['max_axis_displacement_m'],float(max(abs(pos-np.asarray(row['first_position'])))))
            first=np.asarray(row['first_quaternion']);dot=abs(float(quat@first)/(np.linalg.norm(quat)*np.linalg.norm(first)))
            row['max_rotation_deg']=max(row['max_rotation_deg'],float(np.degrees(2*np.arccos(np.clip(dot,0,1)))))
            row['min_z_m']=min(row['min_z_m'],float(pos[2]));row['last_position']=pos.tolist()
            return value
        layout.get_instance_pose=measured
        result=None
        try:
            result=original(*args,**kwargs)
            return result
        finally:
            layout.get_instance_pose=get_pose
            print('OPENETA_STABILITY_DIAGNOSTIC '+json.dumps({'native_result':result,'actors':records},default=str),flush=True)
    checked._openeta_diagnostics=True
    layout.check_layout_stability=checked
