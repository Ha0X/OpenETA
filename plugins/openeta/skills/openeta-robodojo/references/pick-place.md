# Pick and place

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use when the task requires acquiring and relocating a rigid object, sorting, exchanging positions, or placing at an oriented destination. A task may only require pickup or release; do not add unrelated stages. Table-supported pushing can use `push-slide` without a grasp.

## Decisions and observations
- Resolve the target and intended destination from public RGB-D and the task instruction. A depth point is a visible surface sample, not an object pose. Track identity through occlusion and reacquire it when ambiguous.
- Choose a grasp for the intended next action: transport clearance, required object orientation, accessible release, and any receiving hand. A successful empty-hand orientation is not interchangeable after acquiring an asymmetric object.
- When available, compare `gripper_state.aperture.width_m` and `width_limits_m` with the intended grasp region in current RGB-D/wrist views. This is calibrated robot clearance at the TCP plane; openness is joint travel fraction, and 100% open can still be too narrow. Other insertion depths can have different clearance. Consider a narrower region, different jaw direction or approach depth when the visible target cannot enter between the pads. Do not infer object size or a secure grip from the width value alone.
- Before closing, inspect whether the chosen region is actually between the fingers with useful pad overlap, rather than only touching the tips. After a failed lift, reconsider that region and alignment; repeating closure at the same apparent contact may only pinch or push the exterior. These are planning suggestions, not required motion phases.
- Approach with clearance and inspect closure; only legacy profiles require marking the intended finger/object contact. When free-space transport is needed, use a short admissible lift and visible object co-motion to test acquisition. A closed command or table-supported stability is insufficient.
- Transport from the measured state. Separate long travel from constrained descent; account for the carried object's visible extent. At the destination, inspect support and finger escape space before release. In legacy profiles, use the accompanying table-placement contract only when that support really is the table; the generic profile needs no place label or support proof.

- For an asymmetric carried object, placing the TCP over a destination may leave the object on a rim. If a useful reference point on the object is visible, call `mark_point` with `relative_to_tcp` set to the holding role. Supply the returned `receipt.tcp_point.point_in_tcp` as `move_to.target.point_in_tcp`, and set `target.xyz` to that point's desired world location. Omit orientation axes to preserve orientation. This uses a public RGB-D sample and measured robot pose only; it assumes the point stays fixed relative to the hand. Remeasure after slip or regrasp, and inspect the object's remaining extent and finger clearance; the point is not a center-of-mass, support or grasp estimate.
- When the task permits, consider supporting the receiving container before placing another object into it. If it must remain held, choose a reachable receiving pose with a clear opening and wrist escape space before bringing the other hand close. These are strategy options, not required stages or permission checks.

## Progress and recovery
Compare object motion and final placement with the intended effect, not just TCP arrival. If the object stays behind or keeps rotating, safely release/reposition and change the contact region or jaw alignment. If only a path is blocked, revise the route without needlessly squeezing again. Repeated closure can consume many native actions; read settling and budget feedback.

For ordered or memory tasks, maintain an explicit compact record of observed identity, completed placements, and the next required item. Confirmation comes from current evidence or official outcome; historical text does not establish state. Reserve enough actions to release and return the active arms when the instruction requires it.

## Capability boundary
The generic physical-contact profile permits plain finger/environment contact without a task-intent label. It does not provide force control, grasp or stability proof, held-object trajectory prediction or mechanism state sensing. Verified live world-articulation links are monitored only when admitted by the profile; their future motion is not predicted. Use the attached profile contract; successful motion does not establish task success.

### Legacy profile notes (not generic-profile gates)
Basic rigid acquisition and tabletop placement are available. Generic receptacle insertion, force-controlled seating and arbitrary object/object contact permissions are not provided by this guide. Read `stack-support` or `align-insert` when those interactions become necessary.
