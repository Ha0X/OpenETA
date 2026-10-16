# Use a held tool

Workflow advice only; this is not an action gate. Read the attached profile contract for actual execution limits. In physical_contact_v1 use plain motion/gripper commands with no intent authorization. Legacy contact fields apply only to legacy profiles.

## When to use
Use when the robot must acquire a tool and act through its working end: for example a sweep, scrape or tap. If the body can simply be pushed while supported, consider `push-slide` without imposing acquisition and handover first.

## Decisions and observations
- Identify the working surface and intended effect from public images. Choose a grasp that tolerates the likely load and keeps fingers clear of the working end; a free-space hold does not establish loaded stability.
- For handover, plan the receiver's usable working grasp first, then the delivery grasp and presentation. Reobserve both contacts and use actual feedback; avoid creating an unsupported shared-object constraint.
- Mark the visible working point relative to the measured TCP if that aids target geometry. `point_in_tcp` transforms coordinates only; after slip or regrasp it may no longer describe the same tool point.
- Use plain motion for the desired working-end path in the generic profile. Only legacy profiles use the scoped `tool_contact` contract and explicit workpiece selection. Check the working end's height and facing, not only the wrist target.
- After each bounded stroke compare actual workpiece motion with the intended effect, and examine grip drift. Adjust the working face/contact geometry before repeating an ineffective stroke.

## Progress and recovery
If the tool turns in the closed hand under load, change the grasp or presentation; a new travel route alone will not cure it. If material accumulates outside a receiver, inspect receiver orientation and the tool's contact direction. Only legacy profiles expose explicit retained support; in the generic profile inspect actual support visually. Restore support or retreat through checked motion when contact bounds stop execution.

## Capability boundary
The generic physical-contact profile permits plain finger/environment contact without a task-intent label. It does not provide force control, grasp or stability proof, held-object trajectory prediction or mechanism state sensing. Verified live world-articulation links are monitored only when admitted by the profile; their future motion is not predicted. Use the attached profile contract; successful motion does not establish task success.

### Legacy profile notes (not generic-profile gates)
Quasi-static table-supported strokes and small adjustments are available. Controlled impacts, commanded contact velocity, force/torque regulation and generic key/strike-event sensing are not. Do not turn a sweep API into a forceful percussion command. A visible tap might be achievable within existing limits, but xylophone or other impact-dependent tasks remain unvalidated. Report the required missing primitive rather than assuming all tool-use tasks share identical execution dynamics.
