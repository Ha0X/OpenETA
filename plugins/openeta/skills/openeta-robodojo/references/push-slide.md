# Push and slide without grasping

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use when a supported rigid body should translate or rotate under finger pressure, including pushing another body with it. No closure, probe lift, or successful grasp is a prerequisite. Use `tool-use` if the plan actually requires carrying a grasped tool.

## Decisions and observations
- Determine the desired object effect: translation, rotation, or both. Select a visible contact region and direction consistent with support and clearance. Off-centre contact may rotate the body instead of translating it.
- Choose one or both fingers and an approach height from public RGB-D. Consider table and workpiece contact when choosing finger height; keep non-working robot parts clear. An open or closed gripper can be appropriate; change openness only when it benefits the interaction.
- Request the desired motion directly in the generic physical-contact profile. Only legacy profiles use `push_contact` and its accompanying marked-region contract.
- Execute a short segment, then compare the object's visible displacement/orientation with the goal. Remeasure and adjust contact location, stroke direction or hand pose when the body pivots, escapes, or barely moves. A TCP at the endpoint is not evidence of effective pushing.

## Progress and recovery
On a path-envelope stop, reassess the measured hand pose and contact load before retrying. If a finger would hit a workpiece, reposition behind the pushed body or choose another contact geometry. Legacy per-call contact permission must be renewed for withdrawal; the generic profile requires no permission renewal. A backward wrist motion is not a pull unless actual geometry/friction can sustain that interaction.

## Capability boundary
The generic profile accepts orientation changes, waypoints and configured dual-arm requests without a push label. Object motion is not predicted and no force is measured. Short segments and reobservation remain useful strategy. Legacy push_contact remains single-arm, straight, constant-orientation and bounded; those legacy restrictions do not apply to plain generic-profile commands.
