"""Fail before replacing a benchmark's required support robot with dual X5."""
from .layout import task_robot_name


def validate_task_robot(task_name, settings, robot_variant):
    native_robot = task_robot_name(task_name, settings)
    if native_robot != 'dual_x5':
        raise ValueError('Native task requires robot configuration '+str(native_robot)+
                         '; fixed X5 Host has no support-robot adapter')
    if robot_variant not in ('single_x5','dual_x5'):
        raise ValueError('Unsupported X5 execution variant')
    return native_robot
