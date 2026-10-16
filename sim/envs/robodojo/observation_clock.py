"""Capture-time evidence, not an estimate from action counts or wall time."""
import math
import numbers


def capture_clock(native, action_index):
    sim = getattr(native, 'sim', None)
    tick = getattr(sim, '_sim_step_counter', None)
    dt = getattr(sim, 'physics_dt', None)
    valid = (isinstance(tick, numbers.Integral) and not isinstance(tick, bool) and tick >= 0
             and isinstance(dt, numbers.Real) and math.isfinite(dt) and dt > 0)
    return {'status': 'available' if valid else 'unavailable',
            'action_index': int(action_index),
            'physics_step': int(tick) if valid else None,
            'simulation_time_s': float(tick * dt) if valid else None,
            'source': 'native_physics_counter_at_adapter_observation_capture',
            'origin': 'native_simulation_start; not_episode_reset',
            'camera_exposure_time_verified': False,
            'wall_clock_realtime': False}
