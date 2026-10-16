"""Reviewed public requirement names. Never serialize native labels or values."""


def requirement(task, node):
    if task not in {
        'fold_clothes', 'pour_liquid_into_cup', 'pour_by_language', 'pour_balls_into_vase',
        'fasten_screws', 'fill_egg_holder', 'fill_pen_holder', 'insert_key', 'make_toast',
        'swap_blocks', 'store_laptop_and_headphones', 'store_tools_in_toolbox', 'sweep_blocks',
        'put_bottles_into_dustbin', 'arrange_largest_number', 'play_stacking_toy',
    }:
        return None
    if not (isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict)):
        return None  # An AND/OR root must never be blamed on one alternative.
    name, args = node
    labels = tuple(args.get(k) for k in ('label', 'label_A', 'label_B'))
    if any(x is not None and not isinstance(x, str) for x in labels):
        return None
    label, a, b = labels
    if name == 'all_robot_back_to_origin':
        return 'both robot arms returning to their initial poses'
    if name == 'is_all_gripper_open':
        return 'the final gripper-open condition'
    if task == 'fold_clothes' and label == 'target':
        point = args.get('point_A')
        if name.startswith('is_garment_pointA_close_to_pointB_by_'):
            if (point, args.get('point_B')) in {('left_sleeve','right_chest'),('right_sleeve','left_chest')}:
                return 'the sleeves tucked across the chest'
            if (point, args.get('point_B')) in {('left_hem','left_shoulder'),('right_hem','right_shoulder')}:
                return 'the hem folded toward the shoulders'
        if name == 'is_garment_line_intersection_angle_less_than_threshold':
            return 'the alignment of the folded hem and shoulder edges'
    if task in ('pour_liquid_into_cup', 'pour_by_language'):
        if name == 'is_axis_up' and label in ('bottle','bottle_0','bottle_1','bottle_2'):
            return 'the source bottle upright check trigger'
        if name == 'is_A_fluid_in_B':
            return 'the required liquid transfer into the matching receiving container'
    if task == 'pour_balls_into_vase':
        if name == 'is_axis_up' and label == 'cup':return 'the source cup restored upright'
        if name == 'is_A_in_B' and a in {f'sphere_{i}' for i in range(7)} and b == 'vase':
            return 'all balls placed in the vase'
    if task == 'fasten_screws':
        if name == 'is_axis_up' and label in ('nut0','nut1','nut2'):
            return 'the movable nuts kept upright on their fixed bolts'
        if (a,b) in {(f'nut{i}',f'bolt{i}') for i in range(3)}:
            return {'is_AB_xy_distance_within_threshold':'a nut centered on its matching fixed bolt',
                    'is_A_depth_in_B':'a nut seated onto its matching fixed bolt'}.get(name)
    if task == 'fill_egg_holder':
        if a in ('target0','target1','target2','target3') and b == 'egg_holder':
            return {'is_A_in_B':'all eggs placed inside the holder',
                    'is_A_point_above_B_point_by_z_range':'the eggs seated in the lower interior of the holder'}.get(name)
        if name == 'is_joint_position_above_ratio' and label == 'egg_holder':return 'the egg-holder lid closed'
    if task == 'fill_pen_holder':
        if a in ('target0','target1','target2','target3') and b == 'pen_holder':
            return {'is_AB_xy_distance_within_threshold':'each pen aligned inside the holder',
                    'is_A_depth_in_B':'each pen sufficiently inserted into the holder'}.get(name)
        if name == 'is_functional_point_lower_than_root_point' and label in ('target0','target1','target2','target3'):
            return 'the required downward orientation of an inserted pen'
        if name == 'is_axis_up' and label == 'pen_holder':return 'the pen holder upright'
    if task == 'insert_key' and (label == 'key' or a == 'key'):
        return {'is_A_depth_in_B':'the key seated in the keyhole',
                'is_AB_xy_distance_within_threshold':'the key centered in the keyhole',
                'is_axis_up':'the key upright at the lock',
                'is_axis_aligned':'the key final turned orientation relative to the lock'}.get(name)
    if task == 'make_toast':
        if name == 'is_N_A_in_B' and b == 'bread_shelf':return 'the two remaining slices left in the bread rack'
        if name == 'is_joint_position_above_ratio' and label == 'toaster':return 'the toaster lever pressed down'
    if task == 'swap_blocks' and label == 'button0':
        return {'is_joint_position_below_ratio':'the confirmation button pressed after the current transfer',
                'is_joint_position_above_ratio':'the confirmation button fully released after the current transfer'}.get(name)
    if task == 'store_laptop_and_headphones':
        if name == 'is_A_functional_point_close_to_B_functional_point' and (a,b)==('headset','stand'):
            return 'the headphone headband aligned on its supporting cradle'
        if name == 'is_joint_position_above_ratio' and label=='laptop':return 'the laptop closed'
        if name == 'is_axis_up' and label=='laptop':return 'the laptop upright in its storage orientation'
        if name == 'is_A_functional_point_in_B_bbox' and (a,b)==('laptop','vertical_laptop_storage_rack'):
            return 'the laptop bottom seated in the vertical rack'
    if task == 'store_tools_in_toolbox':
        nouns={'hammer':'the hammer','tape_measure':'the tape measure','wrench':'the wrench','pliers':'the pliers'}
        if name=='is_A_functional_point_close_to_B_functional_point' and a in nouns and b=='toolbox':
            return nouns[a] + ' aligned in position and orientation with its matching recess'
        if name=='is_A_cover_B' and a=='toolbox' and b in nouns:
            return nouns[b] + ' fully contained in the toolbox'
    if task == 'sweep_blocks':
        if name=='is_A_on_B_left' and (a,b)==('broom_shovel','broom'):return 'the dustpan left of the broom in the final arrangement'
        if label=='broom_shovel':
            return {'is_not_lift':'the dustpan left at table height',
                    'is_axis_up':'the dustpan level in the final arrangement'}.get(name)
    if task == 'put_bottles_into_dustbin' and name=='is_A_on_B_bottom' and a in {f'bottle{i}' for i in range(4)} and b=='dustbin':
        return 'all bottles fully contained inside the dustbin at the required height'
    # Do not invent visual identities for private per-instance digit/peg labels.
    return None
