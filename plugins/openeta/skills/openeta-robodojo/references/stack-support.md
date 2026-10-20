# Stack and build supports

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use for towers, blocks on blocks, bowls nested together, or another placement whose stability depends on existing objects. Reuse `pick-place` for acquisition and transit; this guide adds support and release reasoning.

## Decisions and observations
- Identify the intended support, its visible top/opening, orientation and usable footprint. Build a stable dependency order from the instruction; do not treat every destination as a flat table.
- Choose a carried orientation and approach that leave clearance for both fingers and their later withdrawal. Resolve large orientation changes above clear space rather than during the final descent.
- Move to an observable pre-placement pose, reobserve, then use bounded permitted motion to approach the support. Inspect whether the object is resting, tilted, caught on a rim, or still suspended by the fingers.
- Plan release and withdrawal separately. Opening can sweep fingers into a support even if the closed-gripper pose is collision-free. After release, withdraw along a visible clear route and inspect whether the assembly remains stable.

## Progress and recovery
If a bowl rim or support catches the object, retreat along an admitted route and reassess alignment; repeating the same descent does not solve the interference. If opening is rejected, inspect the named finger/contact pair and revise release geometry. Regrasp and slightly lift only if the current state and permissions allow it. Do not continue building on an uncertain lower layer.

## Capability boundary
The generic physical-contact profile permits plain finger/environment contact without a task-intent label. It does not provide force control, grasp or stability proof, held-object trajectory prediction or mechanism state sensing. Verified live world-articulation links are monitored only when admitted by the profile; their future motion is not predicted. Use the attached profile contract; successful motion does not establish task success.

### Legacy profile notes (not generic-profile gates)
Some stacking and nesting episodes have succeeded with current checked motion and feedback, but this is not a general stacking capability guarantee. `support_contact` certifies the marked table only; it does not authorize a block, bowl rim or shelf as an arbitrary support. No generic pre-release swept-finger preview or public support-stability estimator is exposed. A contact requiring absent authorization is a capability gap, not a reason to disable checking or repurpose `tool_contact` without meeting its actual contract.
