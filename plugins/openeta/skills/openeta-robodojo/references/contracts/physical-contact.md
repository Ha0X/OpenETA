# Generic rigid physical-contact profile

Applies when live collision_policy.contact_policy is physical_contact_v1.
This describes rigid contact. When combined with cloth_visual_short_horizon_v2,
its accompanying cloth contract additionally limits segments and enforces
synchronized batches; the 1 mm rigid penetration bound is not a cloth-force bound.
Use plain move_to and gripper_control. No intent label, contact mark, support proof,
or grasp-before-motion is required. mark_point remains a public measurement tool.
Rotations, up to four waypoints and configured dual-arm motion use normal schemas.
The Host never searches alternate spatial routes. Failed preview returns before
motion; a runtime stop can leave a partially executed motion. Choose any revised
path yourself. Independent timing adjustments remain allowed by that explicit
coordination mode; use synchronized when timing is part of the intended action.

Only actual finger links may contact environment geometry under this profile.
Other robot/environment and robot/robot pairs keep checked motion protection.
Finger contact is checked at each physics step against a 1 mm geometric
penetration bound, not force. Existing deeper contact may not worsen on a later
request; no automatic retreat occurs. Geometry bound failures can be conservative.
Contact motion is not limited to a push/place/stroke category or 5 cm translation;
short observed segments are advice, not a prerequisite or a required workflow.

No force sensing/control, grasp confirmation, support-stability inference or
held-object trajectory prediction is provided. Environment bodies remain dynamic
world geometry, refreshed for robot collision monitoring; their swept trajectories
and object/object impacts are not certified. A clear preview is not a guarantee
about the result of pushing or transporting an object. Use public images to verify
progress. No hidden object geometry is returned as touch or task perception.

Read workflow guidance if useful. Loading or following a skill is not an execution
gate. Legacy push_contact/support_contact/tool_contact and contact_mark fields
are absent in this profile; never add them to a generic request. Cloth requires
the explicit visual cloth execution contract; this rigid policy alone does not
provide it. Fluids remain unsupported. World articulations require verified per-link physical poses;
check the live collision policy. Future mechanism motion is not predicted.

## Time and fine translation

wait(duration_s) accepts (0, 2] seconds, billed through native actions. It holds
arm positions measured at wait start and preserves latched gripper intent.
Duration rounds up to a native action period; receipts report actual physics time,
early stops and budget. Scene dynamics and native task events continue. No direct
stage advancement is available. Holding position is not force limiting.

For small translations, use target.delta_xyz and frame=world or tcp instead of
xyz. At most 5 cm per delta; orientation is preserved. The first delta uses the
measured TCP; later waypoint deltas use the preceding requested target. Conversion
is explicit in the receipt. Do not combine delta with point_in_tcp or new axes.
The default completion tolerances are 3 mm / 0.03 rad. Where exposed, request
position_tolerance_m from 0.0005 to 0.003 on move_to (at the top level for a
motions batch). The criterion applies to all selected arms and the final target;
intermediate waypoints retain normal tracking tolerance. The Host keeps the same
target and drive gains and reports failure if the requested tolerance is unmet.
It may spend additional bounded native actions tracking that target. This is not
a guarantee that arbitrary poses are reachable at 0.5 mm precision. Completion,
contact, insertion and official task success remain separate evidence.

observation_clock reports the physical counter at adapter capture, with an explicit
origin and availability. It is not a verified camera exposure timestamp or a
wall-clock realtime promise. There is no hidden object velocity, material state,
fluid volume, grasp classifier or force channel in this profile.
