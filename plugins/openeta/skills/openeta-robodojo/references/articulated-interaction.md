# Press, pull, open and close mechanisms

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use when an object is meant to move along a hinge, slider or button travel, often between ordinary pick/place stages. This guide helps identify requirements; it does not enable articulated-object control.

## Decisions and observations
- Identify the visible handle/button/lid, intended state change and likely motion direction. Do not infer a hidden hinge axis, travel limit or latch state from a single point, or request simulator joint metadata unavailable to the agent.
- Distinguish a short contact press from grasped pulling, hinged rotation and sustained support. Choose an approach that allows withdrawal and does not trap the hand when the mechanism changes state.
- Before contact, check whether the live profile exposes permission and feedback for the mechanism. If not, report the specific missing capability; ordinary free-space approach remains only an approach, not an authorized press/pull.
- With an admitted live-link profile, propose ordinary move_to/gripper_control commands from public observations; use their measured robot state/stop feedback and fresh images to verify the state transition. Reacquire geometry after opening/closing before the next placement.

## Progress and recovery
For a permitted button interaction, count a press only from an observed or reported press/release transition, not the number of commands. For a drawer/lid, reaching the hand target does not prove the mechanism moved. If no motion or a stop is observed, reassess direction, engagement and clearance instead of blindly increasing penetration. Do not substitute repeated `observe` for physics advancement when a task awaits a native stage transition.

## Capability boundary
The generic physical-contact profile permits plain finger/environment contact without a task-intent label. It does not provide force control, grasp or stability proof, held-object trajectory prediction or mechanism state sensing. Verified live world-articulation links are monitored only when admitted by the profile; their future motion is not predicted. Use the attached profile contract; successful motion does not establish task success.

### Legacy profile notes (not generic-profile gates)
Legacy checked RoboDojo profiles do not offer a general movable-articulation world, mechanism contact authorization, hinge/slider-constrained motion, force limiting, or explicit wait/stage-advance tool. `push_contact` handles selected supported rigid bodies; it is not a button/drawer contract. `support_contact` is table placement, not lid support. No mechanism action API is invented here. Compose supported `pick-place` or `align-insert` preparation only if it helps the task, and report where a missing mechanism capability prevents completion.

When wait is exposed, use a bounded wait for settling or a native event, then inspect fresh images and public task feedback. Wait is billed and may stop early; it cannot force a stage transition. A live collision link is not a mechanism sensor for the Agent.
