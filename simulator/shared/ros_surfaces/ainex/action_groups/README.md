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

Standing, the hands sweep roughly **0.15 m to 0.45 m above the surface the feet are on**
— the arms are short relative to a 0.40 m robot, so from its init pose it grasps at
chest height, never off the ground. Measured: upright, the claw tip is 0.379 m over the
sole; with the legs folded flat and the torso level it is *still* 0.072 m up. What brings
it to the surface is the torso leaning forward, which the real robot's hip chain does over
planted feet when it crawls, and which here is a joint of its own, `base_pitch`
(`ainex_model`, step 2), because the base rides the torso and folding the hips alone
lifts the legs instead of lowering the body.

So a frame may carry **`base_pitch`** beside `servos`: the torso's lean in radians,
carried forward like every other channel and zero wherever it is not mentioned. The
ground-follow (`ground.py`) keeps the stance sole on the surface while the legs fold under
it, which is what lets the body come down.

`crawl_left` / `crawl_right` — the vendor's names for the bend-down grasp its pick-up
demo runs between `hand_back` and `place_block` — are **solved, not typed**, by
`shared/tools/author_ainex_crawl.py`, which sweeps the lean, the leg fold and one arm
inside the servo limits until the TCP lands an apple's radius above the surface in
front of the feet. Their `reach:` key records where, and
`shared/tests/ainex_grasp_check.py` puts the task's apple there and asserts a hand geom
touches it and it moves. `test_attach.py` holds every other group inside the standing
band and these two below 45 mm.
