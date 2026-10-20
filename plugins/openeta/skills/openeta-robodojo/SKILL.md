---
name: openeta-robodojo
description: Operate a configuration-bound RoboDojo episode with atomic robot tools and choose or combine grasping, stacking, insertion, pushing, tool-use, mechanism, cloth-edge acquisition and moving-target workflows from public observations.
---

# RoboDojo task workflows

Call `episode_status` first to inspect the task, current public images, robot roles, actual tool capabilities and official outcome. Choose an interaction from the instruction and observed geometry. The workflow descriptions below are strategy guidance, not an executable macro, a required grasp-first sequence or evidence that every described capability exists.

## Choose guidance as needed

The restricted episode has no filesystem-reading tool. Call the OpenETA documentation operation `read_skill(skill_id=...)` to read a selected workflow and its applicable contact contracts. It reads only the packaged, hash-bound guides; it cannot read repository code, assets, historical episodes or arbitrary paths. It advances no physics and consumes one tool-request slot, not a native action. Reuse already loaded guidance; change or combine workflows when the task/evidence requires it.

| skill_id | Read when | Reference |
|---|---|---|
| `pick-place` | Acquiring, transporting, sorting or placing a rigid object | [Pick/place](references/pick-place.md) |
| `stack-support` | Building a support relation, stacking or nesting | [Stack/support](references/stack-support.md) |
| `align-insert` | Fitting a slot, peg, hook or shaped destination; includes unsupported-contact boundaries | [Align/insert](references/align-insert.md) |
| `push-slide` | Supported manipulation through finger pressure without requiring grasp or lift | [Push/slide](references/push-slide.md) |
| `tool-use` | Acting through a held tool's working surface; sweep and impact capabilities differ | [Tool use](references/tool-use.md) |
| `articulated-interaction` | Pressing or moving mechanisms; check missing capabilities before interaction | [Mechanisms](references/articulated-interaction.md) |
| `cloth-contact` | Visible cloth edge/corner acquisition, optional scoop-and-pinch and visual probe lift; with a live cloth profile | [Cloth strategy](references/cloth-edge-grasp.md), [legacy contract](references/contracts/cloth.md), [visual v2 contract](references/contracts/cloth-visual.md) |
| `moving-target` | Following public temporal observations, intercepting a moving object, or remembering an earlier appearance | [Moving targets](references/moving-target.md) |

For cloth, use `cloth-contact` when helpful and do not apply rigid attachment/push/tool assumptions. Geometric proximity/following is not force or tactile feedback, and cannot establish grasp. Use permitted public observations and report uncertainty.

The base robot operations are `episode_status`, `observe`, `mark_point`, `move_to`, `gripper_control`, and `finish_episode`. The physical-contact profile also exposes bounded `wait`; use only tools in the live schema. `read_skill` is documentation only. Live schemas and measured receipts define available request fields, permission scope and stops. This skill does not grant unavailable insertion, mechanism, impact, force-control, fluid or dynamic-tracking capabilities. Do not inspect hidden benchmark state or use operator task-source analysis as episode perception.

## Physical-contact profile

When collision_policy.contact_policy is physical_contact_v1, use plain move_to/gripper_control with no push/place/grasp intent or contact authorization. Rotations, waypoints and dual-arm requests are accepted by the generic schemas. Fingers may contact the environment within measured geometric bounds; other robot collisions remain protected. Skills are optional advice, never a precondition. No force control or held-object trajectory prediction is available. Workflow reads include the applicable physical-contact contract rather than legacy authorization instructions. Visual cloth v2 adds per-step cloth safety checks, 5cm/0.25rad first segments and synchronized dual-arm requests; no mesh-based grasp evidence is exposed. These cloth limits override generic segment and timing guidance. Legacy profiles retain their own scoped contracts.

## Shared observation, motion and coordination

Use the displayed original image dimensions when selecting pixels. `mark_point(camera, pixel)` returns a measured visible surface point in world metres. It does not infer the object centre, hidden surface or grasp pose. For a thin object on a table, approach the visible intended contact region carefully rather than descending blindly below its surface. Use a second camera when geometry is ambiguous.

Inspect the returned mark preview before relying on a point. Its gapped yellow crosshair shows the selected pixel in a crop of the same public RGB frame, with crop origin and original pixel coordinates stated explicitly. The preview does not identify the object. If the selected surface is not the intended one, choose a new point; continue to give original-image pixel coordinates, not crop coordinates. A point on the tabletop cannot measure the height or offset of a tool's working end.

`move_to` targets the calibrated grasp TCP, not the wrist link. Each target has `xyz`; optional `approach` and `jaw` are perpendicular world direction vectors. Approach is the TCP +Z direction towards the fingertips; jaw is the TCP +X direction in which fingers separate. Both are required when changing orientation. Omit both to keep orientation. `tcp_state` reports these same axes. An empty parallel gripper has half-turn symmetry: prefer the orientation closest to the current pose when selecting a grasp. While holding an asymmetric object, a half-turn also rotates that object; preserve its intended orientation instead of treating the two poses as interchangeable.

Plan the route, including clearance for the forearm, wrist camera and fingers. First retreat into open space when a large orientation change is needed. `waypoints` accepts up to four intermediate TCP targets; use them to control approach and transport instead of requesting one long diagonal motion. The controller interpolates Cartesian targets, checks IK joint targets and monitors physical execution. A rejected or partially completed motion requires a new plan from the returned measured state. An IK error calls for changing approach, twist or position; repeating the same request does not repair it.

For multiple effectors, choose the arm with a suitable approach and reach. The unselected arm holds its measured starting joints. Submit two independent goals together using `motions`, with one target, waypoints and optional contact mark per role. By default independent coordination tries concurrent execution, a bounded temporal delay, then either sequential order; inspect `coordination`, `actual_timing` and `host_adjustments`. Automatic spatial replanning and joint-space detours are disabled. A failed Cartesian candidate is returned to you without trying another route. Targets and waypoints are preserved. Do not provide start times. Two separate `move_to` calls remain sequential and are never silently batched.

When changing the relative timing would change the task intent, use `coordination="synchronized"` alongside `motions`. Jointly supporting the same object must not use independent retiming; the Host rejects detected shared-body constraints in independent mode. The synchronized path retains existing joint/relative-wrist checks, not a guarantee that every cooperative manipulation is supported. Automatic independent coordination preserves each arm's endpoint and waypoints; it does not release objects, move to a different goal, or perform task strategy changes. A rejected request has not moved; an interrupted request may have partially moved, so replan from the returned state. Gripper commands select one role and preserve the other gripper's command.

If a finger has already contacted the table, request a short upward retreat of at most 5 cm while preserving the current orientation. The controller admits this recovery only when it can prove that every contacting finger vertex moves upward and all other collision checks pass. It does not admit arbitrary motion from a colliding start. The receipt reports recovery mode, joint step limits and any remaining contact.

If a measured positive gap is already below the normal path margin, an ordinary short retreat or gripper adjustment can enter `positive_clearance_retreat`. The Host certifies nonintersecting predicted joint segments, checks that these gaps do not shrink and remain positive after each physics step, and preserves the normal margin for all other pairs. Arm recovery requires a single translation of at most 5 cm with unchanged orientation; independent dual-arm scheduling first requires this boundary to be resolved. Read `recovery_diagnostics` and `host_adjustments`; this is not authorization for robot-to-robot contact or arbitrary close motion.

## Historical experience when available

If the live schema exposes `review_experience`, you may inspect its catalog and request a relevant segment before a difficult approach or recovery. It provides historical before/after RGB, measured TCP/joint/gripper state, recorded requests and feedback. Read segments selectively; images for each segment are sent once. A reviewer may identify a useful partial operation in an unsuccessful episode: inspect both the attributed clip review and the separate official episode outcome. Re-ground targets and contact in the current images; old coordinates, marks, grasp claims and outcomes never become current evidence. This reference does not require you to copy a sequence or prevent a different strategy.

## Progress, sequencing and budget

Use memory/ordering, handover and retained support only when the task needs them. Select workflows by interaction rather than a hard-coded task name. A composite task may alternate free-space transfer, pushing and a mechanism operation; completing one does not authorize the next. `observe` does not advance physics. If the environment requires waiting for a moving target or a native stage, use billed `wait` when exposed, then inspect new observations. If it is unavailable, report that capability gap. Do not send unrelated robot motions as a clock. No tool forces a task-stage transition.

Maintain a small record of the current subgoal, observed progress, pending order and known failure. Task memory is not current geometric evidence. Read `host_adjustments` and plan from the actual partially executed pose after a stop. A failed IK search is not proof of global impossibility. Do not repeat a rejected geometry or an ineffective contact solely because requests remain.

Current experiment defaults provide 200 Host tool calls, with no total native-action or wall-clock cap; check the returned status because historical profiles can retain finite limits. Elapsed time and used native actions are recorded. Reserve tool calls for required return/release and verification. If a finite episode action cap is reported, also reserve native actions based on recent execution costs. Stop unproductive grasp/regrasp loops when the task can use a supported push instead. A motion completion and task completion are separate observations.

Inspect returned images after every motion. `status.simulation_budget` reports native action accounting separately from `remaining_requests`. When `unlimited=true`, `action_limit` and `remaining_actions` are null: there is no episode action cap, and used actions are statistics, not a reason to stop. Finite per-call execution horizons and safety stops still apply; replan from the returned state when needed. A zero-motion preview may explicitly omit images only when the RGB-D and calibration exactly match images already shown; use `observe` whenever another view is needed. After manipulation, when compatible with the task instruction, leave objects stably supported, release tools and retreat unloaded arms toward their observed initial safe poses. Check the official outcome again after cleanup. Trust `status.outcome` for official task success; tool execution success and reaching a pose are separate facts. Call `finish_episode(success=true, reason=...)` only with official success evidence. Otherwise finish with a concise reason when the task cannot continue within the episode budget.
