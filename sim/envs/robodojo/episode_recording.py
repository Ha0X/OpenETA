"""Operator-only RGB recording from existing observations, without extra steps."""
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np


def projection_snapshot(observation):
    """Public calibration/proprioception captured with this RGB, never live/future poses."""
    from copy import deepcopy
    from .scene_geometry import pose_matrix
    result={'schema':'openeta.recorded_tcp_projection.v1','status':'unavailable',
            'source':'same_observation_public_calibration_and_proprioception'}
    try:
        root=observation['proprio']
        entries=root.get('arms') or {'primary':root}
        poses={}
        for role,entry in entries.items():
            metadata=entry['metadata']
            if metadata.get('robot_contract') != 'openeta.x5_parallel_link6_tcp.v1':
                raise ValueError('Unverified calibration')
            pose=np.asarray(entry['ee_pose'],dtype=float)
            tcp=np.asarray(metadata['tcp_from_reported'],dtype=float)
            if pose.shape!=(7,) or tcp.shape!=(4,4) or not np.isfinite(pose).all() or not np.isfinite(tcp).all():
                raise ValueError('Invalid calibrated pose')
            poses[role]=(pose_matrix(pose[[0,1,2,6,3,4,5]]) @ tcp).tolist()
        cameras={}
        for name,frame in observation['cameras'].items():
            rgb=np.asarray(frame['rgb'])
            cameras[name]={'frame_id':name,'width':int(rgb.shape[1]),'height':int(rgb.shape[0]),
                'intrinsics':deepcopy(frame['intrinsics']),'extrinsics':deepcopy(frame['extrinsics'])}
        result.update(status='available',tcp_poses=poses,cameras=cameras)
    except (KeyError,ValueError,TypeError,IndexError):
        result['reason']='Missing or invalid same-frame public robot/camera calibration'
    return result


class EpisodeRecording:
    def __init__(self, root, *, task, seed):
        self.path = Path(root)/('episode-'+uuid.uuid4().hex)
        self.path.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self._physics_origin = None
        self.metadata = {'task':task,'seed':seed,'started_unix_s':time.time(),
            'source':'existing_public_rgb_observations_after_each_native_action',
            'extra_physics_steps':0,'extra_renders':0,'closed':False,
            'tcp_projection':'per_frame_public_proprioception_and_calibration_v1',
            'raw_rgb_preserved':True,
            'time_basis':'simulation; model reasoning pauses omitted',
            'camera_layout':['cam_head','cam_left_wrist','cam_right_wrist']}
        self._write_metadata()

    def _write_metadata(self):
        (self.path/'metadata.json').write_text(json.dumps({**self.metadata,'frames':self.count},indent=2)+'\n')

    def capture(self, adapter):
        from PIL import Image, ImageDraw
        cameras = adapter._last_obs['cameras']
        control = getattr(adapter,'_checked_controller',None)
        cloth = getattr(control,'cloth',None)
        native = getattr(adapter,'_env',None) or getattr(control,'native',None)
        sim = getattr(native,'sim',None)
        step = getattr(sim,'_sim_step_counter',None)
        seconds = None
        if step is not None:
            step = int(step)
            if self._physics_origin is None:
                if adapter._step_count != 0:
                    raise ValueError('Recording physics clock must start at episode reset')
                self._physics_origin = step
            if step < self._physics_origin:
                raise ValueError('Recording physics clock moved backwards')
            seconds = float((step-self._physics_origin)*sim.physics_dt)
            self.metadata['clock_source'] = 'native_physics_step_counter_since_reset'
        elif cloth is not None:
            seconds = float(cloth.step*control.native.sim.physics_dt)
            self.metadata['clock_source'] = 'cloth_physics_step_counter'
        if adapter._step_count == 0:
            seconds = 0.
        canvas=Image.new('RGB',(640,752))
        for name,box in [('cam_head',(0,32,640,480)),('cam_left_wrist',(0,512,320,240)),
                         ('cam_right_wrist',(320,512,320,240))]:
            if name not in cameras:
                raise ValueError('Recording camera missing: '+name)
            rgb=np.asarray(cameras[name]['rgb'])
            frame=Image.fromarray(rgb[:,:,:3]).convert('RGB')
            canvas.paste(frame.resize(box[2:]),box[:2])
        label=f'Action {adapter._step_count} | simulation {seconds:.3f}s' if seconds is not None else f'Action {adapter._step_count} | simulation clock unavailable'
        ImageDraw.Draw(canvas).text((8,9),label,fill='white')
        filename=f'{self.count:06d}.jpg'
        canvas.save(self.path/filename,quality=92)
        with (self.path/'frames.jsonl').open('a') as stream:
            stream.write(json.dumps({'frame':filename,'action_index':adapter._step_count,
                                     'simulation_time_s':seconds,'recorded_unix_s':time.time(),
                                     'tcp_projection':projection_snapshot(adapter._last_obs)})+'\n')
        self.count += 1
        self._write_metadata()

    def close(self):
        self.metadata['closed']=True
        self._write_metadata()


def start_recording(adapter):
    previous=getattr(adapter,'_episode_recording',None)
    if previous is not None:
        previous.close()
    root=os.environ.get('OPENETA_EPISODE_RECORDING_DIR')
    adapter._episode_recording=(EpisodeRecording(root,task=adapter._task_name,seed=adapter._seed) if root else None)
    if adapter._episode_recording is not None:
        adapter._episode_recording.capture(adapter)


def capture_recording(adapter):
    recorder=getattr(adapter,'_episode_recording',None)
    if recorder is not None:
        recorder.capture(adapter)
