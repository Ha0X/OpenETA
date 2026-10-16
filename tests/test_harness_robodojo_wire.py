from copy import deepcopy

import numpy as np
import pytest

from tools.harness_config import ConfigError, ConfigRegistry
from tools.harness_robodojo import official_outcome, camera_directory, inspect_packet
from test_harness_robodojo_adapter import Native, adapter, action


def test_old_success_flag_is_unknown_even_at_terminal_and_reward_one():
    for flags in ({'success':True}, {'task_success':True}, {'benchmark_success':True}):
        result = official_outcome({'ok':True, 'reward':1, 'terminated':True, 'info':flags})
        assert result['status'] == 'unknown' and result['benchmark_success'] is None


@pytest.mark.parametrize('status', ['running','success','failure'])
def test_native_adapter_receipt_roundtrips_into_benchmark_codec(status):
    env = adapter(Native(status))
    env.reset()
    observation,reward,terminated,truncated,info = env.step(action())
    packet = {'observation':observation, 'reward':reward, 'terminated':terminated, 'truncated':truncated, 'info':info}
    result = inspect_packet(ConfigRegistry().resolve('robodojo_dual_x5_v1'), packet)
    assert result['outcome']['status'] == status
    assert not result['movement_authorized']
    assert set(result['robot']['effectors']) == {'left','right'}


def test_inconsistent_or_unversioned_outcome_rejected():
    env = adapter(Native())
    env.reset()
    info = env.step(action())[-1]
    for change in ({'status':'success','benchmark_success':True}, {'terminal':1}, {'action_index':True}, {'source':'unknown'}):
        modified = deepcopy(info)
        modified['official_outcome'].update(change)
        with pytest.raises(ConfigError):
            official_outcome({'info':modified})


def test_raw_camera_matrices_do_not_establish_optical_conventions():
    result = camera_directory({'cameras': {'head': {'intrinsics':np.eye(3), 'extrinsics':np.eye(4), 'depth':[[1.]]}}})
    assert not result['head']['pixel_to_world_available']


def test_explicit_camera_calibration_and_duplicate_names():
    camera = {'frame_id':'head', 'intrinsics':{'fx':10.,'fy':10.,'cx':4.,'cy':4.},
              'extrinsics':{'camera_to_world':np.eye(4).tolist(),'camera_frame':'opencv','reference_frame':'world'},
              'depth_convention':'optical_z_m'}
    assert camera_directory({'cameras':[camera]})['head']['pixel_to_world_available']
    with pytest.raises(ConfigError):
        camera_directory({'cameras':[camera,camera]})
    camera['extrinsics']['camera_to_world'][0][0] = -1
    assert not camera_directory({'cameras':[camera]})['head']['pixel_to_world_available']
