# Align, insert and hang

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use when a carried object's direction and approach must match a slot, socket, peg, hook, or shaped recess. Threading or turning after insertion adds constrained rotation and may require capabilities beyond geometric alignment.

## Decisions and observations
- Identify visible mating features and their approach directions; obtain another view if the opening or orientation is ambiguous. A single surface point cannot measure a hidden hole axis or insertion depth.
- Choose the grasp backward from the mating pose, leaving the active feature and release path unobstructed. Use `pick-place` for acquisition, including handover only when needed and supported.
- Establish a pre-aligned pose outside contact with the receiver. Separate orientation correction from final approach and use short moves based on fresh observations.
- Advance only under an actually exposed contact contract covering the interacting bodies. Inspect alignment, visible entry and returned stops after each segment. Do not equate TCP convergence with seating, thread engagement or successful hanging.

## Progress and recovery
On an obstruction, first distinguish unreachable orientation, unintended receiver collision, and unknown engagement. Back out through an admitted path before a material alignment change. Do not apply blind lateral search, repeated inward pressure or assumed torque. Turning requires evidence of engagement and an appropriate supported rotation constraint.

## Capability boundary
The generic physical-contact profile permits plain finger/environment contact without a task-intent label. It does not provide force control, grasp or stability proof, held-object trajectory prediction or mechanism state sensing. Verified live world-articulation links are monitored only when admitted by the profile; their future motion is not predicted. Use the attached profile contract; successful motion does not establish task success.

### Legacy profile notes (not generic-profile gates)
Current tools support public RGB-D alignment, checked poses, waypoints and grasp contact. They do not expose generic mating-surface authorization, contact-force feedback, impedance control, insertion-depth sensing or screw/lock engagement verification. `contact_mark` permits selected fingers on the selected body; it does not automatically permit the held object to enter another object. `tool_contact` is a scoped table-supported rigid interaction and is not a generic insertion or threaded-joint API. Stop and describe the missing capability when the required contact cannot be admitted.

When exposed, target.delta_xyz with frame=world or tcp expresses short translations without inventing hidden mating coordinates. Keep orientation fixed during that delta; use explicit axes in a separate pose request when needed. The default 3 mm completion tolerance is not insertion-depth evidence. When exposed, request a smaller position_tolerance_m for a fine segment, inspect the achieved error, and stop or revise your proposal if the requirement is unmet. A stricter completion criterion does not create force control or certify engagement.
