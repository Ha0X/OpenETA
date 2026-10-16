# Moving targets and timed interactions

Use this guide when public images show a moving object, conveyor, or a task that requires remembering an earlier appearance. Combine it with the relevant manipulation workflow. This guide grants no target tracker, timing guarantee, force sensor, or collision-check exemption.

## Observe the simulation clock

Read the public `observation_clock`, action receipts, and current tool schema. An observation timestamp describes capture time, not a verified camera exposure time. `observe` does not advance physics. Use bounded `wait(duration_s=...)` to advance the environment when available; it holds the measured arm configuration and preserves gripper intent. Waiting is billed and may be interrupted. Inspect its actual duration: it rounds UP to native action periods.

Do not use network latency, reasoning time, repeated observations of the same physics step, or a remembered benchmark speed as evidence of object velocity. Compare two or more public images at different simulation times. If needed, select corresponding visible features using fresh `mark_point` results and estimate displacement divided by elapsed simulation time. Matching the same object/feature is your visual judgment. Rotation, occlusion, slip, acceleration, and uncertain depth can invalidate the estimate. Private simulator identities, poses, velocities, and reward labels are not available to this workflow.

## Prepare, then use short execution intervals

Choose an accessible interception region from public images and move into a clear approach pose before the object arrives. Preorient and open the gripper when appropriate. An object approaching a reachable region may be easier to intercept than to chase with long diagonal moves. This is optional advice: pushing, guiding, or a different approach can fit the task better.

A short target displacement does not bound how long the controller will spend converging. When exposed, `max_duration_s` on `move_to` and `gripper_control` is a ceiling on simulated execution, not a requested speed, arrival time, or start time. It rounds DOWN to whole native actions; less than one action is rejected. For example, if the reported native period is 0.04 s, a 0.10 s ceiling permits at most 0.08 s. This is an illustrative timebase, not a benchmark constant.

Far from contact, normal moves can be useful. As timing becomes important, use a short ceiling compatible with the available timebase, inspect actual elapsed simulation time and measured progress, and reobserve before the next proposal. Plan for the time required to descend AND close the gripper. A fixed `xyz` does not follow a moving object. Returned `requested_execution_horizon_reached` means the requested observation opportunity was reached; it is not a collision or proof of task failure. Earlier safety, budget, or terminal stops take precedence. The arm trajectory will not automatically resume; submit a new request from current measurements if appropriate. Gripper intent remains latched, so later move/wait calls can continue the prior closure/opening intent. Issue a new gripper command when you want to change it.

For a dual-arm batch, the ceiling covers the entire request, including any retiming/serialization delay. If a timing adjustment would change your task intent, use `coordination="synchronized"`; this does not guarantee exact simultaneous physical arrival. Do not invent per-arm start times.

After closure, inspect measured opening and visual object motion. Closure alone does not establish grasp. If the object passes through or slips, reobserve and reconsider timing, approach, or strategy. Do not keep pursuing stale coordinates or repeat the same ineffective command merely because budget remains.

## Clear incoming traffic before holding or waiting

After visually confirming a grasp, check whether the **whole held object**, including its underside and protrusions, clears the path of incoming objects. A raised TCP or a stationary arm does not establish that clearance. From public camera views and, when useful, fresh depth/`mark_point` observations, choose a higher or lateral holding pose and a clear path that preserves your task intent and leaves room for the other arm. Do not assume a fixed lift distance is sufficient; reobserve to verify both the grasp and visible clearance.

While waiting for another target, consider incoming traffic as well as the target itself. Use short waits and recheck clearance when objects approach or the view is uncertain. Holding joint positions does not keep moving objects away from the held object. An abrupt displacement may indicate an external impact rather than a weak initial grasp; inspect the available views before deciding how to recover. These are planning suggestions, not a tracking guarantee or permission for the Host to alter your path automatically.

## Memory, completion, and stopping

For a remember-and-match instruction, retain the first object's visible distinguishing features and observation order before advancing the scene. Remembering an identity does not preserve its location. For an image-board instruction, inspect the public board and compare candidates visually; never infer a target from internal names.

Use official completion feedback when warranted. Stop an attempt when the target is no longer accessible, observations cannot distinguish candidates, or repeated corrections show the current strategy is ineffective. Report the specific uncertainty or missing capability. Task success is not required to produce a useful diagnostic; do not change scene speed, reset secretly, or request unlimited retries.
