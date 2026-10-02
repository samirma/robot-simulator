# Evidence index

Generated 2026-10-02T11:48:52+01:00 by `tests/evidence.py` (workspace spec §3). One case per robot of `robots_specs/high_level_spec.md` per engine, on the engine's default scene (`simulator/kitchen.sh start --engine <engine>`), spawned with `simulator/spawn.sh` (mobile robots on the floor, arms on the worktop).

**Overall: PASS** (12/12 cases pass)

| Engine | Robot | Scene | Placement | Result | Picture | Cameras | Smoke run | Console check | Attempts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| molmospaces | `myagv` | ithor:1 | floor | PASS | <a href="molmospaces/myagv/scene.png"><img src="molmospaces/myagv/scene.png" width="160"></a> [scene.png](molmospaces/myagv/scene.png) | [cam__usb_cam__image_raw.png](molmospaces/myagv/cam__usb_cam__image_raw.png) | [smoke.json](molmospaces/myagv/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass | [myagv](molmospaces/myagv/fleet_myagv.txt) exit 0 | 2 |
| molmospaces | `so101` | ithor:1 | worktop | PASS | <a href="molmospaces/so101/scene.png"><img src="molmospaces/so101/scene.png" width="160"></a> [scene.png](molmospaces/so101/scene.png) | [cam__image_raw.png](molmospaces/so101/cam__image_raw.png) | [smoke.json](molmospaces/so101/smoke.json): arm pass, gripper pass | [so101](molmospaces/so101/fleet_so101.txt) exit 0 | 2 |
| molmospaces | `ainex` | ithor:1 | floor | PASS | <a href="molmospaces/ainex/scene.png"><img src="molmospaces/ainex/scene.png" width="160"></a> [scene.png](molmospaces/ainex/scene.png) | [cam__camera__image_raw.png](molmospaces/ainex/cam__camera__image_raw.png) | [smoke.json](molmospaces/ainex/smoke.json): head_pan pass, head_tilt pass, walk pass, action_group pass | [ainex](molmospaces/ainex/fleet_ainex.txt) exit 0 | 3 |
| molmospaces | `mycobot280` | ithor:1 | worktop | PASS | <a href="molmospaces/mycobot280/scene.png"><img src="molmospaces/mycobot280/scene.png" width="160"></a> [scene.png](molmospaces/mycobot280/scene.png) | none recorded | [smoke.json](molmospaces/mycobot280/smoke.json): arm pass, gripper pass | [mycobot280](molmospaces/mycobot280/fleet_mycobot280.txt) exit 0 | 1 |
| molmospaces | `myagv_mycobot280` | ithor:1 | floor | PASS | <a href="molmospaces/myagv_mycobot280/scene.png"><img src="molmospaces/myagv_mycobot280/scene.png" width="160"></a> [scene.png](molmospaces/myagv_mycobot280/scene.png) | [cam__usb_cam__image_raw.png](molmospaces/myagv_mycobot280/cam__usb_cam__image_raw.png) | [smoke.json](molmospaces/myagv_mycobot280/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass, arm pass, gripper pass | [myagv](molmospaces/myagv_mycobot280/fleet_myagv.txt) exit 0<br>[mycobot280](molmospaces/myagv_mycobot280/fleet_mycobot280.txt) exit 0 | 1 |
| molmospaces | `rosmaster_x3_plus` | ithor:1 | floor | PASS | <a href="molmospaces/rosmaster_x3_plus/scene.png"><img src="molmospaces/rosmaster_x3_plus/scene.png" width="160"></a> [scene.png](molmospaces/rosmaster_x3_plus/scene.png) | [cam__camera__rgb__image_raw.png](molmospaces/rosmaster_x3_plus/cam__camera__rgb__image_raw.png)<br>[cam__camera__depth__image_raw.png](molmospaces/rosmaster_x3_plus/cam__camera__depth__image_raw.png)<br>[cam__camera__ir__image_raw.png](molmospaces/rosmaster_x3_plus/cam__camera__ir__image_raw.png) | [smoke.json](molmospaces/rosmaster_x3_plus/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass, arm pass, gripper pass | [rosmaster_x3_plus](molmospaces/rosmaster_x3_plus/fleet_rosmaster_x3_plus.txt) exit 0 | 2 |
| robocasa | `myagv` | robocasa:1-1 | floor | PASS | <a href="robocasa/myagv/scene.png"><img src="robocasa/myagv/scene.png" width="160"></a> [scene.png](robocasa/myagv/scene.png) | [cam__usb_cam__image_raw.png](robocasa/myagv/cam__usb_cam__image_raw.png) | [smoke.json](robocasa/myagv/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass | [myagv](robocasa/myagv/fleet_myagv.txt) exit 0 | 1 |
| robocasa | `so101` | robocasa:1-1 | worktop | PASS | <a href="robocasa/so101/scene.png"><img src="robocasa/so101/scene.png" width="160"></a> [scene.png](robocasa/so101/scene.png) | [cam__image_raw.png](robocasa/so101/cam__image_raw.png) | [smoke.json](robocasa/so101/smoke.json): arm pass, gripper pass | [so101](robocasa/so101/fleet_so101.txt) exit 0 | 1 |
| robocasa | `ainex` | robocasa:1-1 | floor | PASS | <a href="robocasa/ainex/scene.png"><img src="robocasa/ainex/scene.png" width="160"></a> [scene.png](robocasa/ainex/scene.png) | [cam__camera__image_raw.png](robocasa/ainex/cam__camera__image_raw.png) | [smoke.json](robocasa/ainex/smoke.json): head_pan pass, head_tilt pass, walk pass, action_group pass | [ainex](robocasa/ainex/fleet_ainex.txt) exit 0 | 1 |
| robocasa | `mycobot280` | robocasa:1-1 | worktop | PASS | <a href="robocasa/mycobot280/scene.png"><img src="robocasa/mycobot280/scene.png" width="160"></a> [scene.png](robocasa/mycobot280/scene.png) | none recorded | [smoke.json](robocasa/mycobot280/smoke.json): arm pass, gripper pass | [mycobot280](robocasa/mycobot280/fleet_mycobot280.txt) exit 0 | 1 |
| robocasa | `myagv_mycobot280` | robocasa:1-1 | floor | PASS | <a href="robocasa/myagv_mycobot280/scene.png"><img src="robocasa/myagv_mycobot280/scene.png" width="160"></a> [scene.png](robocasa/myagv_mycobot280/scene.png) | [cam__usb_cam__image_raw.png](robocasa/myagv_mycobot280/cam__usb_cam__image_raw.png) | [smoke.json](robocasa/myagv_mycobot280/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass, arm pass, gripper pass | [myagv](robocasa/myagv_mycobot280/fleet_myagv.txt) exit 0<br>[mycobot280](robocasa/myagv_mycobot280/fleet_mycobot280.txt) exit 0 | 1 |
| robocasa | `rosmaster_x3_plus` | robocasa:1-1 | floor | PASS | <a href="robocasa/rosmaster_x3_plus/scene.png"><img src="robocasa/rosmaster_x3_plus/scene.png" width="160"></a> [scene.png](robocasa/rosmaster_x3_plus/scene.png) | [cam__camera__rgb__image_raw.png](robocasa/rosmaster_x3_plus/cam__camera__rgb__image_raw.png)<br>[cam__camera__depth__image_raw.png](robocasa/rosmaster_x3_plus/cam__camera__depth__image_raw.png)<br>[cam__camera__ir__image_raw.png](robocasa/rosmaster_x3_plus/cam__camera__ir__image_raw.png) | [smoke.json](robocasa/rosmaster_x3_plus/smoke.json): drive_forward pass, drive_back pass, drive_left pass, drive_right pass, drive_turn pass, arm pass, gripper pass | [rosmaster_x3_plus](robocasa/rosmaster_x3_plus/fleet_rosmaster_x3_plus.txt) exit 0 | 2 |

## How a case is judged

* **Scene picture**: an offscreen 1280x720 render through the simulation's control port, from a viewpoint with a clear line of sight to the robot (checked with a depth render).
* **Camera pictures**: one frame of every camera not marked `optional` in the robot's interface file (`robots_specs/<id>/ros*.yml`), taken off the vendor wire through rosbridge and decoded from its `sensor_msgs/Image` (depth colourised: near red, far blue, no return black); its encoding, size and frame id must be the recorded ones.
* **Smoke run** (`smoke.json`): each motion uses the robot's recorded command (interface file `motions[].command` and its example) on the vendor wire; a drive or walk is followed by the recorded stop. A motion passes when the recorded feedback (or, where the interface records none, the control port's joint/pose readings) shows the commanded displacement within the tolerance recorded in the interface file -- a `{relative, absolute}` pair bounds |measured - commanded| by max(relative x |commanded|, absolute) -- and the robot is at rest afterwards. Each motion in `smoke.json` names its bound and rule. Base motions: 0.25 m forward/back/left/right at 0.2 m/s and a 1 rad turn at 0.5 rad/s, within the placement's guaranteed travel.
* **Console check**: `robot_console/python.sh -m robot_console.fleet --url <wire> --expect <profile>` for every console profile naming the robot or one of its components; exit 0 is the console's own verdict.
* A case also requires `spawn.sh` to exit 0 on SIGINT with no container left.
* **Worktop objects** (an arm on the worktop): the spawn staged the six worktop objects (apple, plate, bowl, mug, banana, lemon) around the arm (simulator spec §2.3); the scene picture frames them with the arm, and the case records them and any scene objects cleared for them.

## Cases

### molmospaces / myagv — PASS

* Readiness: `spawn ready: myagv (myAGV) in molmospaces ithor:1 on the floor; wire(s): myagv ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-02T11:20:34+01:00, finished 2026-10-02T11:26:38+01:00; host load (1 min) before 6.87; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](molmospaces/myagv/case.json), [smoke.json](molmospaces/myagv/smoke.json), [spawn.log](molmospaces/myagv/spawn.log), [simulation.log](molmospaces/myagv/simulation.log)
* Earlier attempt 2026-10-02T11:18:39+01:00: FAIL (load 5.56, RTF warnings 7, RTF below 0.90 during collection: [{'window_end': '2026-10-02T11:19:18+01:00', 'rtf': 0.461}, {'window_end': '2026-10-02T11:19:29+01:00', 'rtf': 0.601}, {'window_end': '2026-10-02T11:19:38+01:00', 'rtf': 0.55}, {'window_end': '2026-10-02T11:19:49+01:00', 'rtf': 0.518}], checks {'scene': True, 'cameras': True, 'fleet': True, 'smoke': False, 'spawn_clean_exit': True}) — [molmospaces/myagv/history/2026-10-02T111839+0100](molmospaces/myagv/history/2026-10-02T111839+0100/case.json)

<img src="molmospaces/myagv/scene.png" width="240"> <img src="molmospaces/myagv/cam__usb_cam__image_raw.png" width="240">

Motions (before / after):

* `drive_back`: <img src="molmospaces/myagv/motion_drive_back_before.png" width="200"> <img src="molmospaces/myagv/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="molmospaces/myagv/motion_drive_forward_before.png" width="200"> <img src="molmospaces/myagv/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="molmospaces/myagv/motion_drive_left_before.png" width="200"> <img src="molmospaces/myagv/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="molmospaces/myagv/motion_drive_right_before.png" width="200"> <img src="molmospaces/myagv/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="molmospaces/myagv/motion_drive_turn_before.png" width="200"> <img src="molmospaces/myagv/motion_drive_turn_after.png" width="200">

### molmospaces / so101 — PASS

* Readiness: `spawn ready: so101 (SO-101) in molmospaces ithor:1 on the worktop; wire(s): so101 ws://127.0.0.1:9090 [ROS 2 jazzy]`
* Worktop objects staged: apple, banana, bowl, lemon, mug, plate; scene objects cleared: `apple_038a0ea9b393da66a161da588e6ecc2a_1_0_0`, `bread_5f50f53c9dcfbae4352335033a8b2bb4_1_0_0`, `book_be545aaaeb1f643586d7965f7a571f70_1_0_0`
* Started 2026-10-02T11:27:57+01:00, finished 2026-10-02T11:33:01+01:00; host load (1 min) before 6.31; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True, 'worktop_objects': True}
* Details: [case.json](molmospaces/so101/case.json), [smoke.json](molmospaces/so101/smoke.json), [spawn.log](molmospaces/so101/spawn.log), [simulation.log](molmospaces/so101/simulation.log)
* Earlier attempt 2026-10-02T11:26:38+01:00: PASS (load 5.54, RTF warnings 3, RTF below 0.90 during collection: [{'window_end': '2026-10-02T11:27:07+01:00', 'rtf': 0.88}, {'window_end': '2026-10-02T11:27:17+01:00', 'rtf': 0.854}], checks {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True, 'worktop_objects': True}) — [molmospaces/so101/history/2026-10-02T112638+0100](molmospaces/so101/history/2026-10-02T112638+0100/case.json)

<img src="molmospaces/so101/scene.png" width="240"> <img src="molmospaces/so101/cam__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="molmospaces/so101/motion_arm_before.png" width="200"> <img src="molmospaces/so101/motion_arm_after.png" width="200">
* `gripper`: <img src="molmospaces/so101/motion_gripper_before.png" width="200"> <img src="molmospaces/so101/motion_gripper_after.png" width="200">

### molmospaces / ainex — PASS

* Readiness: `spawn ready: ainex (AiNex) in molmospaces ithor:1 on the floor; wire(s): ainex ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-02T11:33:01+01:00, finished 2026-10-02T11:34:02+01:00; host load (1 min) before 4.45; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](molmospaces/ainex/case.json), [smoke.json](molmospaces/ainex/smoke.json), [spawn.log](molmospaces/ainex/spawn.log), [simulation.log](molmospaces/ainex/simulation.log)
* Earlier attempt 2026-10-02T10:56:31+01:00: PASS (load 6.69, RTF warnings 3, RTF below 0.90 during collection: [{'window_end': '2026-10-02T11:00:10+01:00', 'rtf': 0.49}, {'window_end': '2026-10-02T11:00:19+01:00', 'rtf': 0.888}], checks {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}) — [molmospaces/ainex/history/2026-10-02T105631+0100](molmospaces/ainex/history/2026-10-02T105631+0100/case.json)
* Earlier attempt 2026-10-02T11:01:09+01:00: FAIL (load 6.65, RTF warnings 0, RTF below 0.90 during collection: no, checks {'scene': False, 'cameras': True, 'fleet': False, 'smoke': False, 'spawn_clean_exit': False}, error RuntimeError('port 9080 or 9090 is taken')) — [molmospaces/ainex/history/2026-10-02T110109+0100](molmospaces/ainex/history/2026-10-02T110109+0100/case.json)

<img src="molmospaces/ainex/scene.png" width="240"> <img src="molmospaces/ainex/cam__camera__image_raw.png" width="240">

Motions (before / after):

* `action_group`: <img src="molmospaces/ainex/motion_action_group_before.png" width="200"> <img src="molmospaces/ainex/motion_action_group_after.png" width="200">
* `head_pan`: <img src="molmospaces/ainex/motion_head_pan_before.png" width="200"> <img src="molmospaces/ainex/motion_head_pan_after.png" width="200">
* `head_tilt`: <img src="molmospaces/ainex/motion_head_tilt_before.png" width="200"> <img src="molmospaces/ainex/motion_head_tilt_after.png" width="200">
* `walk`: <img src="molmospaces/ainex/motion_walk_before.png" width="200"> <img src="molmospaces/ainex/motion_walk_after.png" width="200">

### molmospaces / mycobot280 — PASS

* Readiness: `spawn ready: mycobot280 (myCobot 280) in molmospaces ithor:1 on the worktop; wire(s): mycobot280 ws://127.0.0.1:9090 [ROS 2 humble]`
* Worktop objects staged: apple, banana, bowl, lemon, mug, plate; scene objects cleared: `apple_038a0ea9b393da66a161da588e6ecc2a_1_0_0`, `bread_5f50f53c9dcfbae4352335033a8b2bb4_1_0_0`, `book_be545aaaeb1f643586d7965f7a571f70_1_0_0`
* Started 2026-10-02T11:34:03+01:00, finished 2026-10-02T11:34:50+01:00; host load (1 min) before 6.28; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True, 'worktop_objects': True}
* Details: [case.json](molmospaces/mycobot280/case.json), [smoke.json](molmospaces/mycobot280/smoke.json), [spawn.log](molmospaces/mycobot280/spawn.log), [simulation.log](molmospaces/mycobot280/simulation.log)

<img src="molmospaces/mycobot280/scene.png" width="240">

Motions (before / after):

* `arm`: <img src="molmospaces/mycobot280/motion_arm_before.png" width="200"> <img src="molmospaces/mycobot280/motion_arm_after.png" width="200">
* `gripper`: <img src="molmospaces/mycobot280/motion_gripper_before.png" width="200"> <img src="molmospaces/mycobot280/motion_gripper_after.png" width="200">

### molmospaces / myagv_mycobot280 — PASS

* Readiness: `spawn ready: myagv_mycobot280 (myAGV + myCobot 280) in molmospaces ithor:1 on the floor; wire(s): base (myagv) ws://127.0.0.1:9090 [ROS 1 noetic]; arm (mycobot280) ws://127.0.0.1:9091 [ROS 2 humble]`
* Started 2026-10-02T11:34:50+01:00, finished 2026-10-02T11:35:59+01:00; host load (1 min) before 5.34; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](molmospaces/myagv_mycobot280/case.json), [smoke.json](molmospaces/myagv_mycobot280/smoke.json), [spawn.log](molmospaces/myagv_mycobot280/spawn.log), [simulation.log](molmospaces/myagv_mycobot280/simulation.log)

<img src="molmospaces/myagv_mycobot280/scene.png" width="240"> <img src="molmospaces/myagv_mycobot280/cam__usb_cam__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="molmospaces/myagv_mycobot280/motion_arm_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_arm_after.png" width="200">
* `drive_back`: <img src="molmospaces/myagv_mycobot280/motion_drive_back_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="molmospaces/myagv_mycobot280/motion_drive_forward_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="molmospaces/myagv_mycobot280/motion_drive_left_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="molmospaces/myagv_mycobot280/motion_drive_right_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="molmospaces/myagv_mycobot280/motion_drive_turn_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_drive_turn_after.png" width="200">
* `gripper`: <img src="molmospaces/myagv_mycobot280/motion_gripper_before.png" width="200"> <img src="molmospaces/myagv_mycobot280/motion_gripper_after.png" width="200">

### molmospaces / rosmaster_x3_plus — PASS

* Readiness: `spawn ready: rosmaster_x3_plus (ROSMASTER X3 PLUS) in molmospaces ithor:1 on the floor; wire(s): rosmaster_x3_plus ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-02T11:38:56+01:00, finished 2026-10-02T11:44:54+01:00; host load (1 min) before 6.91; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](molmospaces/rosmaster_x3_plus/case.json), [smoke.json](molmospaces/rosmaster_x3_plus/smoke.json), [spawn.log](molmospaces/rosmaster_x3_plus/spawn.log), [simulation.log](molmospaces/rosmaster_x3_plus/simulation.log)
* Earlier attempt 2026-10-02T11:36:00+01:00: PASS (load 5.82, RTF warnings 8, RTF below 0.90 during collection: [{'window_end': '2026-10-02T11:37:18+01:00', 'rtf': 0.466}, {'window_end': '2026-10-02T11:37:29+01:00', 'rtf': 0.337}, {'window_end': '2026-10-02T11:37:39+01:00', 'rtf': 0.446}, {'window_end': '2026-10-02T11:37:49+01:00', 'rtf': 0.61}, {'window_end': '2026-10-02T11:37:59+01:00', 'rtf': 0.571}, {'window_end': '2026-10-02T11:38:09+01:00', 'rtf': 0.456}], checks {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}) — [molmospaces/rosmaster_x3_plus/history/2026-10-02T113600+0100](molmospaces/rosmaster_x3_plus/history/2026-10-02T113600+0100/case.json)

<img src="molmospaces/rosmaster_x3_plus/scene.png" width="240"> <img src="molmospaces/rosmaster_x3_plus/cam__camera__rgb__image_raw.png" width="240"> <img src="molmospaces/rosmaster_x3_plus/cam__camera__depth__image_raw.png" width="240"> <img src="molmospaces/rosmaster_x3_plus/cam__camera__ir__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="molmospaces/rosmaster_x3_plus/motion_arm_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_arm_after.png" width="200">
* `drive_back`: <img src="molmospaces/rosmaster_x3_plus/motion_drive_back_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="molmospaces/rosmaster_x3_plus/motion_drive_forward_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="molmospaces/rosmaster_x3_plus/motion_drive_left_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="molmospaces/rosmaster_x3_plus/motion_drive_right_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="molmospaces/rosmaster_x3_plus/motion_drive_turn_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_drive_turn_after.png" width="200">
* `gripper`: <img src="molmospaces/rosmaster_x3_plus/motion_gripper_before.png" width="200"> <img src="molmospaces/rosmaster_x3_plus/motion_gripper_after.png" width="200">

### robocasa / myagv — PASS

* Readiness: `spawn ready: myagv (myAGV) in robocasa robocasa:1-1 on the floor; wire(s): myagv ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-02T11:44:54+01:00, finished 2026-10-02T11:46:47+01:00; host load (1 min) before 6.17; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](robocasa/myagv/case.json), [smoke.json](robocasa/myagv/smoke.json), [spawn.log](robocasa/myagv/spawn.log), [simulation.log](robocasa/myagv/simulation.log)

<img src="robocasa/myagv/scene.png" width="240"> <img src="robocasa/myagv/cam__usb_cam__image_raw.png" width="240">

Motions (before / after):

* `drive_back`: <img src="robocasa/myagv/motion_drive_back_before.png" width="200"> <img src="robocasa/myagv/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="robocasa/myagv/motion_drive_forward_before.png" width="200"> <img src="robocasa/myagv/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="robocasa/myagv/motion_drive_left_before.png" width="200"> <img src="robocasa/myagv/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="robocasa/myagv/motion_drive_right_before.png" width="200"> <img src="robocasa/myagv/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="robocasa/myagv/motion_drive_turn_before.png" width="200"> <img src="robocasa/myagv/motion_drive_turn_after.png" width="200">

### robocasa / so101 — PASS

* Readiness: `spawn ready: so101 (SO-101) in robocasa robocasa:1-1 on the worktop; wire(s): so101 ws://127.0.0.1:9090 [ROS 2 jazzy]`
* Worktop objects staged: apple, banana, bowl, lemon, mug, plate; scene objects cleared: `rc_obj_2_main`, `rc_obj_3_main`, `rc_obj_6_main`, `rc_obj_8_main`, `rc_obj_9_main`, `rc_obj_10_main`
* Started 2026-10-02T11:46:47+01:00, finished 2026-10-02T11:48:52+01:00; host load (1 min) before 6.32; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 0
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True, 'worktop_objects': True}
* Details: [case.json](robocasa/so101/case.json), [smoke.json](robocasa/so101/smoke.json), [spawn.log](robocasa/so101/spawn.log), [simulation.log](robocasa/so101/simulation.log)

<img src="robocasa/so101/scene.png" width="240"> <img src="robocasa/so101/cam__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="robocasa/so101/motion_arm_before.png" width="200"> <img src="robocasa/so101/motion_arm_after.png" width="200">
* `gripper`: <img src="robocasa/so101/motion_gripper_before.png" width="200"> <img src="robocasa/so101/motion_gripper_after.png" width="200">

### robocasa / ainex — PASS

* Readiness: `spawn ready: ainex (AiNex) in robocasa robocasa:1-1 on the floor; wire(s): ainex ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-02T11:03:30+01:00, finished 2026-10-02T11:05:17+01:00; host load (1 min) before 5.92; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 1
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](robocasa/ainex/case.json), [smoke.json](robocasa/ainex/smoke.json), [spawn.log](robocasa/ainex/spawn.log), [simulation.log](robocasa/ainex/simulation.log)

<img src="robocasa/ainex/scene.png" width="240"> <img src="robocasa/ainex/cam__camera__image_raw.png" width="240">

Motions (before / after):

* `action_group`: <img src="robocasa/ainex/motion_action_group_before.png" width="200"> <img src="robocasa/ainex/motion_action_group_after.png" width="200">
* `head_pan`: <img src="robocasa/ainex/motion_head_pan_before.png" width="200"> <img src="robocasa/ainex/motion_head_pan_after.png" width="200">
* `head_tilt`: <img src="robocasa/ainex/motion_head_tilt_before.png" width="200"> <img src="robocasa/ainex/motion_head_tilt_after.png" width="200">
* `walk`: <img src="robocasa/ainex/motion_walk_before.png" width="200"> <img src="robocasa/ainex/motion_walk_after.png" width="200">

### robocasa / mycobot280 — PASS

* Readiness: `spawn ready: mycobot280 (myCobot 280) in robocasa robocasa:1-1 on the worktop; wire(s): mycobot280 ws://127.0.0.1:9090 [ROS 2 humble]`
* Worktop objects staged: apple, banana, bowl, lemon, mug, plate; scene objects cleared: none
* Started 2026-10-02T10:39:08+01:00, finished 2026-10-02T10:40:01+01:00; host load (1 min) before 6.41; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 1
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True, 'worktop_objects': True}
* Details: [case.json](robocasa/mycobot280/case.json), [smoke.json](robocasa/mycobot280/smoke.json), [spawn.log](robocasa/mycobot280/spawn.log), [simulation.log](robocasa/mycobot280/simulation.log)

<img src="robocasa/mycobot280/scene.png" width="240">

Motions (before / after):

* `arm`: <img src="robocasa/mycobot280/motion_arm_before.png" width="200"> <img src="robocasa/mycobot280/motion_arm_after.png" width="200">
* `gripper`: <img src="robocasa/mycobot280/motion_gripper_before.png" width="200"> <img src="robocasa/mycobot280/motion_gripper_after.png" width="200">

### robocasa / myagv_mycobot280 — PASS

* Readiness: `spawn ready: myagv_mycobot280 (myAGV + myCobot 280) in robocasa robocasa:1-1 on the floor; wire(s): base (myagv) ws://127.0.0.1:9090 [ROS 1 noetic]; arm (mycobot280) ws://127.0.0.1:9091 [ROS 2 humble]`
* Started 2026-10-01T23:12:16+01:00, finished 2026-10-01T23:13:18+01:00; host load (1 min) before 5.51; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 1
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](robocasa/myagv_mycobot280/case.json), [smoke.json](robocasa/myagv_mycobot280/smoke.json), [spawn.log](robocasa/myagv_mycobot280/spawn.log), [simulation.log](robocasa/myagv_mycobot280/simulation.log)

<img src="robocasa/myagv_mycobot280/scene.png" width="240"> <img src="robocasa/myagv_mycobot280/cam__usb_cam__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="robocasa/myagv_mycobot280/motion_arm_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_arm_after.png" width="200">
* `drive_back`: <img src="robocasa/myagv_mycobot280/motion_drive_back_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="robocasa/myagv_mycobot280/motion_drive_forward_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="robocasa/myagv_mycobot280/motion_drive_left_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="robocasa/myagv_mycobot280/motion_drive_right_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="robocasa/myagv_mycobot280/motion_drive_turn_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_drive_turn_after.png" width="200">
* `gripper`: <img src="robocasa/myagv_mycobot280/motion_gripper_before.png" width="200"> <img src="robocasa/myagv_mycobot280/motion_gripper_after.png" width="200">

### robocasa / rosmaster_x3_plus — PASS

* Readiness: `spawn ready: rosmaster_x3_plus (ROSMASTER X3 PLUS) in robocasa robocasa:1-1 on the floor; wire(s): rosmaster_x3_plus ws://127.0.0.1:9090 [ROS 1 noetic]`
* Started 2026-10-01T23:15:12+01:00, finished 2026-10-01T23:16:42+01:00; host load (1 min) before 5.61; real-time factor during the collection: never below 0.90; RTF warnings in the whole simulation log 1
* Checks: {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}
* Details: [case.json](robocasa/rosmaster_x3_plus/case.json), [smoke.json](robocasa/rosmaster_x3_plus/smoke.json), [spawn.log](robocasa/rosmaster_x3_plus/spawn.log), [simulation.log](robocasa/rosmaster_x3_plus/simulation.log)
* Earlier attempt 2026-10-01T23:13:19+01:00: PASS (load 6.29, RTF warnings 3, RTF below 0.90 during collection: [{'window_end': '2026-10-01T23:13:57+01:00', 'rtf': 0.829}], checks {'scene': True, 'cameras': True, 'fleet': True, 'smoke': True, 'spawn_clean_exit': True}) — [robocasa/rosmaster_x3_plus/history/2026-10-01T231319+0100](robocasa/rosmaster_x3_plus/history/2026-10-01T231319+0100/case.json)

<img src="robocasa/rosmaster_x3_plus/scene.png" width="240"> <img src="robocasa/rosmaster_x3_plus/cam__camera__rgb__image_raw.png" width="240"> <img src="robocasa/rosmaster_x3_plus/cam__camera__depth__image_raw.png" width="240"> <img src="robocasa/rosmaster_x3_plus/cam__camera__ir__image_raw.png" width="240">

Motions (before / after):

* `arm`: <img src="robocasa/rosmaster_x3_plus/motion_arm_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_arm_after.png" width="200">
* `drive_back`: <img src="robocasa/rosmaster_x3_plus/motion_drive_back_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_drive_back_after.png" width="200">
* `drive_forward`: <img src="robocasa/rosmaster_x3_plus/motion_drive_forward_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_drive_forward_after.png" width="200">
* `drive_left`: <img src="robocasa/rosmaster_x3_plus/motion_drive_left_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_drive_left_after.png" width="200">
* `drive_right`: <img src="robocasa/rosmaster_x3_plus/motion_drive_right_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_drive_right_after.png" width="200">
* `drive_turn`: <img src="robocasa/rosmaster_x3_plus/motion_drive_turn_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_drive_turn_after.png" width="200">
* `gripper`: <img src="robocasa/rosmaster_x3_plus/motion_gripper_before.png" width="200"> <img src="robocasa/rosmaster_x3_plus/motion_gripper_after.png" width="200">

