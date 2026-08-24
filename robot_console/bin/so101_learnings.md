Learnings from operating this SO-101 rig:

- Every move_joints call MUST carry a `targets` object of "name": value pairs, e.g.
  {"targets": {"0": 15, "1": -40, "5": 0}, "note": "..."}. A call with only a note, or
  with a stringified/singular "target", is rejected and wastes a turn.
- Joint meanings on this arm: 0 = base rotation (turns the whole arm left/right),
  1 = shoulder lift (more negative = arm lower toward the table), 2 = elbow bend,
  3 = wrist pitch, 4 = wrist roll, 5 = gripper where 100 is fully open and 0 is fully
  closed. To grasp an apple-sized object, open to ~90 first, then close to ~5-15.
- Move decisively. Targets tens of degrees away are safe: the rig interpolates
  smoothly and clamps limits below you. Many tiny nudges burn the call budget.
- The main failure mode on this rig is endless alignment: if two corrections in the
  same direction still leave the target on the same side of the gripper, triple the
  step instead of nudging again.
- Close early rather than perfectly. When the fingers appear level with the object
  and on both sides of it, close the gripper immediately (5 -> about 8). A slightly
  imperfect close often still captures it, and you can reopen and retry; running out
  of calls while aligning cannot be recovered.
- Decide the joint-0 image direction ONCE, from your first base rotation, write it in
  a note, and never re-derive it mid-episode: flip-flopping on that direction has
  wasted more call budget than any other mistake. Beware that at large base angles the
  arm's 3D rotation projects non-monotonically into the image -- judge by whether the
  gripper-to-object gap shrank, not by absolute position.
- When the gripper appears within about one apple-width of the object, stop adjusting
  sideways: descend (joint 1 more negative) and close. At close range the camera
  cannot resolve small misalignment, so proximity beats pixel-perfect alignment.
- Reliable pick-and-place: open the gripper; rotate/extend until the gripper is
  directly above the object; lower (joint 1 more negative, adjust 2/3) until the
  fingers straddle it; close the gripper; lift (joint 1 up); move over the container;
  open the gripper.
- POSITIVE joint 2 folds the forearm back toward the base -- an arm with joint 2 above
  ~+20 curls under itself and its gripper hovers near the base no matter what the
  other joints do. To actually extend out to an object ~20-30 cm away on the surface,
  use approximately: joint 1 = -60 to -80, joint 2 = -20 to 0 (extended), joint 3 =
  -50 to -70 (fingers pointing down). Verify extension in the image: the gripper
  should appear clearly separated from the arm's base, most of the way to the object.
- Grasp from directly above with the wrist pitched down (joint 3 strongly negative),
  not from the side: a top-down grasp tolerates a couple of centimetres of error,
  while a side-on grasp pushes the object away.
- If the same grasp misses twice at the same spot, the error is along the camera's
  depth axis, which you cannot see directly: change the approach (different base
  angle, more top-down) instead of repeating the same close.
- The wrist camera view moves with the arm; judge positions relative to fixed objects
  on the counter, not relative to the image frame.
