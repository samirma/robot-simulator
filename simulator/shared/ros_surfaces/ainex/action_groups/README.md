# Action groups

The AiNex has no inverse-kinematics service: on the real robot every manipulation is a
recorded servo trajectory replayed by `MotionManager.run_action`, triggered over ROS by
`/app/set_action`. These are the simulator's stand-ins.

**None of Hiwonder's own action groups are here.** They are vendor pose data with no
stated licence, and the set that circulates publicly is a mirror of an SD-card image.
`actions.py` reads their `.d6a` format anyway, so pointing `--action-dir` at a real
robot's `/home/ubuntu/software/ainex_controller/ActionGroups` plays the genuine motions
and a file there shadows the one here by name.

Format: a list of frames, each a `duration` in seconds and a sparse `servos` map of
**joint name -> radians**. Radians rather than servo counts, following the vendor's own
`init_pose.yaml`, so a pose can be read against that file directly. Joints a frame does
not mention carry forward from the frame before, starting from the init pose — which is
what lets `clamp_left` say "close the claw" without restating the other 23 joints.

## Reach

The torso rides planar joints and **cannot pitch**, so this robot cannot bend forward.
Combined with arms that are short relative to its 0.46 m height, that puts the hands
between roughly **0.25 m and 0.43 m above the floor** — it grasps from a surface at its
own chest height, never off the ground. `test_attach.py` checks the poses below stay in
that band. This is a consequence of the planar base; see `robots/README.md`.
