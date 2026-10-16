from types import SimpleNamespace as S
from sim.envs.robodojo.observation_clock import capture_clock


def test_clock_uses_physics_not_action_count():
    c=capture_clock(S(sim=S(_sim_step_counter=120,physics_dt=.004)),3)
    assert c['physics_step']==120 and c['simulation_time_s']==.48 and c['action_index']==3
    assert not c['camera_exposure_time_verified'] and not c['wall_clock_realtime']


def test_missing_or_invalid_clock_never_estimated():
    for native in [S(),S(sim=S(_sim_step_counter=-1,physics_dt=.004)),S(sim=S(_sim_step_counter=1,physics_dt=float('nan')))]:
        c=capture_clock(native,25)
        assert c['status']=='unavailable' and c['simulation_time_s'] is None
