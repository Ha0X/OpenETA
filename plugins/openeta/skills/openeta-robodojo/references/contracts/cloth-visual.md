# Visual cloth execution contract (v2)

This applies only to `cloth_visual_short_horizon_v2`. The current tool schema and receipts define capabilities.

Use ordinary `move_to`, `gripper_control`, and bounded `wait`. Finger–cloth interaction needs no intent label or contact authorization. Contact with rigid surfaces still has the reported physical-contact penetration limit. Arm links, self collisions and inter-arm collisions remain checked. No automatic spatial detour is taken.

Every move executes at most the first 5 cm / 0.25 rad segment, using a common fraction for both arms. Remaining waypoints are not executed. Read `host_adjustments`, `reached_execution_target`, and `continuation_required`; a completed short segment is not completion of the full requested move. All two-arm requests keep synchronized timing because shared cloth constraints are not measured. Synchronization does not guarantee constant cloth tension or a rigid constraint.

Judge engagement and deformation from returned images. Joint opening is available, but closure is not grasp proof. There is no mesh-derived proximity, following, grasp-loss signal, force or tactile feedback. Private cloth geometry is used only to check nonworking robot parts and other obstacles at each physical step. This does not predict cloth deformation across an entire move.

The worker does not advance simulation merely while waiting for model input. Within an executed segment the agent has no intermediate visual feedback; shorten the next request when interaction is uncertain. The edge-scoop skill is advisory and cannot guarantee a grasp. Use a small visible probe lift and compare images.

If the receipt identifies a modified friction profile, interpret results as that experiment's physics condition. It affects cloth contacts including table sliding, not exclusively finger grip. It does not establish that the cloth was grasped.
