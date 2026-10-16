"""Narrow, idempotent compatibility hooks for the pinned native RoboDojo runtime.

Do not alter reward, terminal flags, rotation composition or support trajectories.
"""


def require_method(owner, name):
    value = getattr(owner, name, None)
    if not callable(value):
        raise RuntimeError('Incompatible RoboDojo runtime: missing '+name)
    return value


def install_robot_naming(robot_manager_class):
    original = require_method(robot_manager_class, 'restore_name')
    if getattr(original, '_openeta_single_arm_fix', False):
        return

    def restore_name(self, processed_name):
        if processed_name == 'ee_joint_state':
            return 'ee'
        return original(self, processed_name)

    restore_name._openeta_single_arm_fix = True
    robot_manager_class.restore_name = restore_name


def install_action_dimensions(eval_module, variant):
    require_method(eval_module, 'get_robot_action_dim_info')
    if variant not in ('single_x5', 'dual_x5'):
        raise ValueError('No checked action dimensions for this execution layout')
    count = 2 if variant == 'dual_x5' else 1
    eval_module.get_robot_action_dim_info = lambda env_cfg: {'arm_dim':[6]*count, 'ee_dim':[1]*count}


def install_action_codec(cls):
    original_validate = require_method(cls, 'validate_action_dict')
    original_type = require_method(cls, 'get_action_type')
    if cls.__dict__.get('_openeta_ee_action_patch', False):
        return

    def validate(self, action):
        if 'arm_ee_pose' in action and 'ee_pose' not in action:
            action = dict(action)
            action['ee_pose'] = action.pop('arm_ee_pose')
        return original_validate(self, action)

    def action_type(self, action):
        if any(key in action for key in ('arm_ee_pose', 'delta_ee_pose', 'left_ee_pose', 'right_ee_pose')):
            return 'ee'
        return original_type(self, action)

    cls.validate_action_dict = validate
    cls.get_action_type = action_type
    cls._openeta_ee_action_patch = True
