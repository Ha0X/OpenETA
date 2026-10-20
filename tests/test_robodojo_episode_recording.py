import json
import numpy as np
from types import SimpleNamespace as NS
from sim.envs.robodojo.episode_recording import start_recording, capture_recording


def test_opt_in_recording_preserves_observation_and_simulation_clock(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENETA_EPISODE_RECORDING_DIR',str(tmp_path))
    rgb=np.full((32,32,3),123,dtype=np.uint8)
    a=NS(_task_name='test',_seed=0,_step_count=0,
         _last_obs={'cameras':{n:{'rgb':rgb} for n in ('cam_head','cam_left_wrist','cam_right_wrist')}})
    start_recording(a)
    a._step_count=1;a._checked_controller=NS(cloth=NS(step=2),native=NS(sim=NS(physics_dt=.004)))
    capture_recording(a);a._episode_recording.close()
    p=a._episode_recording.path
    frames=[json.loads(x) for x in (p/'frames.jsonl').read_text().splitlines()]
    assert [f['simulation_time_s'] for f in frames]==[0.,.008]
    assert all((p/f['frame']).exists() for f in frames)
    assert json.loads((p/'metadata.json').read_text())['closed']
    assert np.all(rgb==123)


def test_recording_disabled_by_default(monkeypatch):
    monkeypatch.delenv('OPENETA_EPISODE_RECORDING_DIR',raising=False)
    a=NS();start_recording(a);capture_recording(a)
    assert a._episode_recording is None


def test_rigid_recording_uses_actual_substeps_after_reset(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENETA_EPISODE_RECORDING_DIR', str(tmp_path))
    rgb=np.zeros((8,8,3),dtype=np.uint8)
    sim=NS(_sim_step_counter=123,physics_dt=.004)
    a=NS(_task_name='rigid',_seed=0,_step_count=0,_env=NS(sim=sim),
         _last_obs={'cameras':{n:{'rgb':rgb} for n in ('cam_head','cam_left_wrist','cam_right_wrist')}})
    start_recording(a)
    # A stopped action advances only three substeps, followed by a zero-step action.
    a._step_count=1;sim._sim_step_counter=126;capture_recording(a)
    a._step_count=2;capture_recording(a)
    rows=[json.loads(x) for x in (a._episode_recording.path/'frames.jsonl').read_text().splitlines()]
    assert [r['simulation_time_s'] for r in rows]==[0.,.012,.012]
    assert sim._sim_step_counter==126


def calibrated_observation():
    cameras={}
    for name in ('cam_head','cam_left_wrist','cam_right_wrist'):
        cameras[name]={'rgb':np.full((480,640,3),80,dtype=np.uint8),
            'intrinsics':{'fx':200.,'fy':200.,'cx':320.,'cy':240.,'width':640,'height':480},
            'extrinsics':{'camera_to_world':np.eye(4).tolist(),'camera_frame':'opencv','reference_frame':'world'}}
    arms={r:{'ee_pose':np.array([x,0,1,0,0,0,1]),'metadata':{
        'robot_contract':'openeta.x5_parallel_link6_tcp.v1','tcp_from_reported':np.eye(4).tolist()}}
        for r,x in [('left',-.2),('right',.2)]}
    return {'cameras':cameras,'proprio':{'arms':arms}}


def test_projection_uses_captured_proprioception_not_controller(tmp_path,monkeypatch):
    from sim.envs.robodojo.episode_recording import projection_snapshot
    obs=calibrated_observation();snap=projection_snapshot(obs)
    assert snap['status']=='available' and snap['tcp_poses']['left'][0][3]==-.2
    obs['proprio']['arms']['left']['ee_pose'][0]=.3
    assert snap['tcp_poses']['left'][0][3]==-.2
    assert projection_snapshot(obs)['tcp_poses']['left'][0][3]==.3
    monkeypatch.setenv('OPENETA_EPISODE_RECORDING_DIR',str(tmp_path))
    a=NS(_task_name='test',_seed=0,_step_count=0,_last_obs=obs)
    start_recording(a);a._episode_recording.close()
    row=json.loads((a._episode_recording.path/'frames.jsonl').read_text())
    assert row['tcp_projection']['tcp_poses']['left'][0][3]==.3
    assert (a._episode_recording.path/row['frame']).exists()


def test_overlay_preserves_raw_and_tracks_each_frame():
    from sim.envs.robodojo.episode_recording import projection_snapshot
    from scripts.render_robodojo_recording import tcp_view
    from PIL import Image
    import copy
    raw=Image.new('RGB',(640,752),'#555555');before=raw.tobytes()
    obs=calibrated_observation();row={'action_index':0,'tcp_projection':projection_snapshot(obs)}
    view,a=tcp_view(raw,row)
    assert a['status']=='available' and all(x['status']=='available' for x in a['cameras'])
    assert raw.tobytes()==before and view.tobytes()!=before
    obs['proprio']['arms']['left']['ee_pose'][0]=.3
    second,b=tcp_view(raw,{'action_index':1,'tcp_projection':projection_snapshot(obs)})
    assert second.tobytes()!=view.tobytes()
    missing,report=tcp_view(raw,{'action_index':2})
    assert report['status']=='unavailable' and raw.tobytes()==before


def test_old_recording_still_exports_raw(tmp_path):
    from scripts.render_robodojo_recording import render
    from PIL import Image
    Image.new('RGB',(64,64),'#333333').save(tmp_path/'0.jpg')
    (tmp_path/'frames.jsonl').write_text(json.dumps({'frame':'0.jpg','action_index':0,'simulation_time_s':0})+'\n')
    result=render(tmp_path,tmp_path/'old.mp4',fps=2)
    assert result['view']=='raw' and (tmp_path/'old.mp4').stat().st_size>0
