# Cloth acquisition from visible edges

Strategy guidance for a live cloth profile (`cloth_short_horizon_v1` or `cloth_visual_short_horizon_v2`). This is a candidate technique to test, not a validated success recipe or required pick/place sequence. Read the accompanying cloth contract for available actions and contact limits. A rigid or GPU-only profile does not acquire cloth capability by reading this guide.

## Choose accessible material

Use public RGB-D to look for a free corner, an overhanging hem, a raised edge or a visible fold that the fingers can straddle. Prefer material with visible entry clearance over pinching the middle of a sheet flat against the table. A depth sample on a thin edge may hit the table or background; inspect its marked crop and another available view before treating it as a cloth point. Do not infer the hidden underside from a single surface depth sample.

Choose the desired material layer from the visible task state. A folded edge may contain several layers; closing on all of them can prevent the intended fold. In legacy v1, use its surface/local-layers authorization deliberately; in visual v2, ordinary finger motion needs no material authorization. Neither contract measures which layers were grasped.

## An optional scoop-and-pinch approach

When the visible geometry supports it, align an open gripper with the edge so one fingertip can pass under an already raised or overhanging portion while the other remains above it. Approach from outside the cloth footprint in a short segment, observe, then make a short inward/upward motion to bring material into the jaw gap. Close after the images support that placement. This is a proposed path for the Agent to choose; the Host does not generate an unseen underside trajectory.

A table-flush edge may have no room for a finger. Do not drive below the observed table plane or repeatedly deepen a rejected scoop. An alternative candidate is a short, permitted tangential contact that gathers a visible wrinkle, followed by reobservation and a new pinch; select another edge if it only drags the garment or the current contact policy rejects the interaction. Existing rigid `push_contact`/`tool_contact` modes do not apply to cloth. No skill grants table collision permission or changes the finger geometry.

Use segment sizes compatible with observed clearance, depth uncertainty and the live completion tolerances. A requested millimetre displacement is not proof it executed, particularly if it is smaller than the active endpoint tolerance. Read actual TCP movement and any partial-execution feedback before planning from the new pose. Use precision and relative-motion options only if present in the current tool schema; visual v2 exposes them, while legacy v1 does not.

## Verify acquisition through a small visible effect

After closure, propose a modest probe lift that keeps most of the garment supported and limits the consequence of an empty grasp. Compare fresh images with the pre-lift view: does the chosen edge rise with the hand, is fabric visibly inside the jaw gap, and does the cloth drape consistently with that contact? Gripper closure or a moving TCP alone is not grasp evidence. A single cloth layer can leave the measured jaw opening almost indistinguishable from an empty closure: near-closed does not prove empty, just as a nonzero opening does not prove held. Resolve this ambiguity through the visible probe effect rather than a rigid-object aperture threshold. If occluded, use another available view or explicitly remain uncertain; do not replace missing visual evidence with mesh proximity/following fields.

If the fingers rise empty or the material slides out, stop the planned transport. Use the visible failure to change the selected edge, jaw alignment, openness or approach geometry. Opening/retracting can be a recovery choice after checking the current supported state and tool feedback; do not blindly repeat an identical close/lift loop. Once a small lift is visually supported, continue in short segments and inspect drape/slip. Two hands manipulating the same cloth keep the existing synchronized execution contract.

## Release and withdrawal

After opening, inspect whether material has visibly separated from both fingers before a large withdrawal. Fabric can remain caught or draped over a finger even when the gripper reports open. Prefer a short withdrawal toward visible free space, observe whether the laid section moves, then continue or revise the direction. If it drags or unfolds, pause and choose a small release/repositioning action from the current images. An automatic home move is not evidence of a clean release. These are recovery suggestions, not a required folding sequence or permission to bypass collision feedback.

If TCP overlays are available, they project robot coordinates, not cloth edges or grasp state. Compare the original images; an overlay crossing the cloth does not establish contact or attachment.

## Feedback timing

The current dedicated RoboDojo worker advances physics through explicit simulator operations; its idle request queue does not continuously step the world while the model reasons. Network waiting therefore does not by itself equal the same amount of simulated cloth motion. Use reported simulation time/physics steps to assess how much the scene advanced, and do not generalize this property to real robots or other servers.

The key blind interval here is the executed segment before the next image. Keep it short enough that a failed acquisition has a limited effect, then observe before committing to a long lift or fold. Per-physics-step Host collision checks are safety checks, not high-frequency visual grasp recognition. Force/tactile sensing is not a prerequisite for trying this visual strategy, but neither this guide nor sparse images guarantee a stable grasp or measured contact load.
