"""The SO-101 arm's ROS contract: joint trajectories in, joint states and cameras out.

    console -> robot   /joint_trajectory_controller/joint_trajectory  trajectory_msgs/msg/JointTrajectory
    console -> robot   /gripper_controller/commands                   std_msgs/msg/Float64MultiArray
    robot -> console   /joint_states                                  sensor_msgs/msg/JointState
    robot -> console   /free_joint_publisher/free_joint_states        mujoco_ros2_control_msgs/msg/FreeJointStateArray
    robot -> console   /overhead/color/compressed                     sensor_msgs/msg/CompressedImage
    robot -> console   /side/color/compressed                         sensor_msgs/msg/CompressedImage
    robot -> console   /wrist/color/compressed                        sensor_msgs/msg/CompressedImage   (opt-in)
    console -> robot   /reset, /mujoco_ros2_control_node/reset_world  Trigger-shaped

This is the topic set a real ros2_control bringup for this arm presents -- a
`joint_trajectory_controller` on the five arm joints, a `forward_command_controller` on
the jaw, one `joint_state_broadcaster` covering both -- so one client drives the arm
here or on a real ROS 2 stack without changing a line. It lives in `shared/` for the
same reason `myagv.py` does: the topic set belongs to the robot, not to an engine, and
two copies would be two chances for the engines to drift far enough apart that a console
could tell them apart.

**The gripper is a topic, not an action, and that is a hard constraint.** The client's
ROS adapter has no action client at all, so a `GripperActionController` -- which is what
the stock SO-ARM controller config declares -- is simply undrivable from it. Publishing
`Float64MultiArray` to a `ForwardCommandController` is the shape that works.

What it needs from an engine is deliberately small: a robot view exposing `arm` and
`gripper` move groups with `joint_pos` / `joint_vel` / writable `ctrl`, the raw model,
and a task object. Everything engine-specific -- how a kitchen gets compiled, where the
arm is mounted -- stays on the engine's side of that line.
"""

from __future__ import annotations

import os
import sys
import threading

import numpy as np

# --------------------------------------------------------------------------- contract

#: Contract joint order. Used for actions, `joint_pos`, and IK on the client. Note the
#: `_joint` suffix: the MJCF calls these `shoulder_pan` and friends, and the translation
#: happens here rather than in the model, so `shared/robots/so101/model.xml` can stay
#: mujoco_menagerie-shaped and the console can stay ROS-shaped.
ARM_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
)
GRIPPER_JOINT = "gripper_joint"
JOINT_ORDER: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)

#: The jaw hinge runs -0.174533..1.745329 rad in the MJCF and 0..1 in the contract, and
#: the map between them is an exact **offset**, not a rescale: `ros2_so_arm` -- the
#: description the contract was written against -- baked the -0.174533 into the jaw
#: body's quaternion and then truncated the range at 1.0. Verified by sweeping the jaw
#: and measuring tip separation: 7.01 / 20.91 / 38.42 / 55.80 mm at contract g =
#: 0.00 / 0.24 / 0.48 / 0.72, against the curve the grasp tuning was fitted on
#: (`gap(g) = -23.24g^2 + 92.54g - 0.84` mm), which predicts 20.86 / 38.29 / 55.79.
#: Agreement to ~0.1 mm, so every measured gripper constant carries over unchanged --
#: including the non-monotonic grasp window, where 0.42..0.52 lifts the apple and 0.55
#: never does. Rescaling instead of offsetting would move the aperture at every value
#: and quietly invalidate all of it.
GRIPPER_OFFSET_RAD = 0.174533
GRIPPER_CONTRACT_RANGE = (0.0, 1.0)

TOPIC_ARM_COMMAND = "/joint_trajectory_controller/joint_trajectory"
TOPIC_GRIPPER_COMMAND = "/gripper_controller/commands"
TOPIC_JOINT_STATES = "/joint_states"
#: Namespaced by the publishing plugin on the reference rig; the name is part of the
#: contract, not an accident, so it keeps the namespace here too.
TOPIC_FREE_JOINT_STATES = "/free_joint_publisher/free_joint_states"
# There is deliberately no `/task_success`. The simulator used to answer the task's own
# question on that topic, and nothing a camera can see corresponds to it: it is the
# scene's private state, a real SO-101 bringup publishes no such thing, and a policy
# graded by it is graded on something outside the robot's senses. The verdict is computed
# from the overhead camera instead -- `robot_console.arm.vision_success`, validated frame
# by frame against the geometry this topic used to report before it was removed.
#
# The free-joint poses stay. They mirror a topic `mujoco_ros2_control` really does
# publish, nothing grades on them any more, and keeping them is what lets the camera
# verdict go on being audited against ground truth instead of being taken on trust.

SERVICE_RESET = "/reset"
SERVICE_RESET_WORLD = "/mujoco_ros2_control_node/reset_world"

#: Camera topic -> the name the MJCF gives that camera, and the frame size to render.
#: Sizes are contract terms: the two scene views are 640x480 because the VLA's
#: preprocessor stretches to 4:3 without preserving aspect, so a 16:9 frame arrives
#: distorted relative to everything it was trained on; the wrist view is 256x256 so a
#: policy resizing to 224 downsamples rather than upsamples.
DEFAULT_CAMERAS: dict[str, tuple[str, int, int]] = {
    "/overhead/color/compressed": ("overhead", 640, 480),
    "/side/color/compressed": ("side", 640, 480),
}
#: The wrist view renders `wrist_cam` -- mujoco_menagerie's own camera, at its published
#: pose and intrinsics, because the SO-101 here is exclusively menagerie's model. It used
#: to render a project-added camera 119 mm and 53.7 degrees away, sited by ray-casting to
#: dodge the wrist_roll_follower housing; that camera is gone along with every other
#: non-upstream edit, and the housing occlusion it avoided is now simply what the official
#: pose sees.
#:
#: 640x360 keeps the camera's own 16:9 aspect (sensorsize 0.00576 x 0.00324 is exactly
#: 16:9) rather than its declared 1920x1080, which is a 2 Mpx render inside the physics
#: loop -- 30x the pixels of the view this replaces, and render cost is what sets the
#: control rate. The aspect is the part a client can be wrong about; the pixel count is a
#: capture mode, which is a thing a real UVC module is configured for too.
WRIST_CAMERA: dict[str, tuple[str, int, int]] = {
    "/wrist/color/compressed": ("wrist_cam", 640, 360),
}


def to_contract_gripper(mjcf_rad: float) -> float:
    """MJCF jaw angle (rad) -> contract 0..1."""
    low, high = GRIPPER_CONTRACT_RANGE
    return float(np.clip(mjcf_rad + GRIPPER_OFFSET_RAD, low, high))


def to_mjcf_gripper(contract: float) -> float:
    """Contract 0..1 -> MJCF jaw angle (rad)."""
    low, high = GRIPPER_CONTRACT_RANGE
    return float(np.clip(contract, low, high)) - GRIPPER_OFFSET_RAD


def attach_ros(
    bus,
    view,
    model,
    task=None,
    *,
    cameras: dict[str, tuple[str, int, int]] | None = None,
    jpeg_quality: int = 70,
    control_hz: float = 10.0,
    scene_option=None,
    world_reset=None,
):
    """Wire the arm onto an already-built bus and return a per-step callback.

    Call the returned function with a `mujoco.MjData` each control period, and with
    `None` to close this robot's streams -- the same shape as `ros_surfaces/myagv.py`.
    It does not stop the server: the fleet that owns the port does that, once.

    The topic constants above stay bare and the `bus` applies the namespace, so
    `/joint_states` reaches the wire as `/so101/joint_states` without a prefix being
    spelled out anywhere in this file.
    """
    from contracts.rosbridge_server import (
        TYPE_FLOAT64_MULTI_ARRAY,
        TYPE_FREE_JOINT_STATE_ARRAY,
        TYPE_JOINT_STATE,
        TYPE_JOINT_TRAJECTORY,
        free_joint_state_array,
        joint_state,
    )
    from mujoco_bridge import CameraStreams
    # Either a MolmoSpaces `RobotView` or a plain mapping of move groups, because that is
    # the one place the two engines genuinely differ: MolmoSpaces builds its groups out of
    # the upstream `RobotView` trio, RoboCasa builds equivalent ones straight off the raw
    # model. Both expose `joint_pos`, `joint_vel` and a writable `ctrl`, which is all this
    # surface has ever needed -- so the difference is absorbed in two lines here rather
    # than by giving one engine its own copy of the contract.
    groups = view if isinstance(view, dict) else {
        gid: view.get_move_group(gid) for gid in ("arm", "gripper")
    }
    arm, gripper = groups["arm"], groups["gripper"]

    # The commanded target, in contract units. Seeded from the pose the engine spawned
    # the arm in rather than from zero: these are position actuators, so a target of
    # zero would drive the arm out of its rest pose the instant the server started, and
    # a client connecting a few seconds later would find an arm that had already
    # collapsed onto the worktop.
    target = np.concatenate(
        [
            np.asarray(arm.joint_pos, dtype=np.float64).reshape(-1),
            [to_contract_gripper(float(np.asarray(gripper.joint_pos).reshape(-1)[0]))],
        ]
    )

    def on_arm_command(msg: dict) -> None:
        """Take the LAST point of a JointTrajectory, matched by joint name.

        Only the endpoint matters: the position servos interpolate, and honouring
        `time_from_start` here would be re-implementing a trajectory controller badly.
        Matching by name rather than by index is the same rule that makes `/joint_states`
        safe to read -- the client is free to send its five joints in any order.
        """
        points = msg.get("points") or []
        names = msg.get("joint_names") or []
        if not points or not names:
            return
        positions = (points[-1] or {}).get("positions") or []
        for name, value in zip(names, positions):
            if name in ARM_JOINTS:
                target[ARM_JOINTS.index(name)] = float(value)

    def on_gripper_command(msg: dict) -> None:
        data = msg.get("data") or []
        if data:
            target[-1] = float(data[0])

    bus.on(TOPIC_ARM_COMMAND, on_arm_command, TYPE_JOINT_TRAJECTORY)
    bus.on(TOPIC_GRIPPER_COMMAND, on_gripper_command, TYPE_FLOAT64_MULTI_ARRAY)
    # `serve_rosapi()` used to be called here. It is a *server* concern -- one topic list
    # answers for every robot on the port -- so the fleet calls it once instead. That it
    # lived here is why a myAGV-only run had no topic discovery at all: the mobile-base
    # surface never called it.

    # A reset is *requested* here and *performed* on the simulation thread. Service
    # handlers run on the websocket thread, and writing qpos from there while the step
    # loop is inside mj_step corrupts the state -- it took the server down with a bare
    # segfault, no traceback, about four seconds into an episode, which is exactly when
    # the client calls /reset. The handler blocks until the loop has applied it, so the
    # reply still means "done" rather than "requested".
    reset_requested = threading.Event()
    reset_applied = threading.Event()

    def do_reset(_args: dict) -> dict:
        """Restore the world to the state the episode should start from.

        Even though this reply is only sent after the reset has been applied, the client
        still polls for the apple to be *measured* back at its spawn pose: on real
        ros2_control stacks the equivalent service answers before the world has moved,
        and the client's check also re-solves the plan from the measured pose, which
        catches a scene that reset fine and is still unreachable.
        """
        reset_applied.clear()
        reset_requested.set()
        if not reset_applied.wait(timeout=5.0):
            return {"success": False, "message": "reset requested but the simulation loop did not apply it"}
        return {"success": True, "message": "world reset"}

    bus.service(SERVICE_RESET, do_reset)
    bus.service(SERVICE_RESET_WORLD, do_reset)

    streams = CameraStreams(
        model, DEFAULT_CAMERAS if cameras is None else cameras, jpeg_quality, scene_option,
        frame_of=bus.frame,
    )

    # `do_reset` runs on a websocket thread and needs the MjData the step loop owns.
    _live: list = [None]

    # SO101_ROS_DEBUG_CONTACTS=1 prints, whenever it changes, how many contacts the jaw
    # geoms have with the task's objects. A grasp that "closes on nothing" is invisible
    # from the wire -- the joint reads its commanded width either way -- and this is the
    # one-line answer to "did the fingers ever touch it".
    contact_debug = os.environ.get("SO101_ROS_DEBUG_CONTACTS") == "1" and task is not None
    jaw_geoms: set[int] = set()
    task_bodies: frozenset[int] = frozenset()
    if contact_debug:
        import mujoco

        task_bodies = task.contact_bodies()
        jaw_geoms = {
            g for g in range(model.ngeom)
            if "jaw" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
        }
    last_contacts = [-1]

    def step(data):
        if data is None:
            streams.close()
            return

        _live[0] = data

        if reset_requested.is_set():
            reset_requested.clear()
            if task is not None:
                task.reset(data)
            target[:5] = np.asarray(arm.joint_pos, dtype=np.float64).reshape(-1)
            target[-1] = to_contract_gripper(float(np.asarray(gripper.joint_pos).reshape(-1)[0]))
            # The task's reset restores the WHOLE world -- qpos, qvel and ctrl -- which on
            # a shared scene includes every other robot in it. Their surfaces latch state
            # the snapshot cannot restore (an integrated `cmd_vel` setpoint, most of all),
            # so they are told to drop it here, inside the same critical section, before
            # the blocked service caller is released.
            if world_reset is not None:
                world_reset.fire()
            reset_applied.set()

        arm.ctrl = target[:5].tolist()
        gripper.ctrl = [to_mjcf_gripper(float(target[-1]))]

        # SIMULATED time, never the wall clock. The success predicate holds for >= 1.0 s
        # of simulated time, the client refuses to start if simulated time is not
        # advancing, and the offline scorer re-derives the hold from these stamps -- all
        # three read this number, and a wall-clock stamp makes all three agree on an
        # answer about the wrong clock.
        stamp = float(data.time)
        seq = bus.next_seq()

        positions = np.concatenate(
            [
                np.asarray(arm.joint_pos, dtype=np.float64).reshape(-1),
                [to_contract_gripper(float(np.asarray(gripper.joint_pos).reshape(-1)[0]))],
            ]
        )
        velocities = np.concatenate(
            [
                np.asarray(arm.joint_vel, dtype=np.float64).reshape(-1),
                np.asarray(gripper.joint_vel, dtype=np.float64).reshape(-1),
            ]
        )
        bus.publish(
            TOPIC_JOINT_STATES,
            joint_state(list(JOINT_ORDER), positions, velocities, stamp),
            TYPE_JOINT_STATE,
        )

        if task is not None:
            bus.publish(
                TOPIC_FREE_JOINT_STATES,
                free_joint_state_array(task.free_joint_entries(data), stamp),
                TYPE_FREE_JOINT_STATE_ARRAY,
            )

        streams.publish(bus, data, seq, stamp)

        if contact_debug:
            n = 0
            for i in range(data.ncon):
                c = data.contact[i]
                a, b = c.geom1, c.geom2
                if (a in jaw_geoms and model.geom_bodyid[b] in task_bodies) or (
                    b in jaw_geoms and model.geom_bodyid[a] in task_bodies
                ):
                    n += 1
            if n != last_contacts[0]:
                last_contacts[0] = n
                print(f"[contacts] t={stamp:.2f} jaw<->task contacts: {n}  gripper={target[-1]:.2f}",
                      file=sys.stderr)

    return step


def serve_ros(
    port: int,
    view,
    model,
    task=None,
    *,
    cameras: dict[str, tuple[str, int, int]] | None = None,
    jpeg_quality: int = 70,
    control_hz: float = 10.0,
    scene_option=None,
    host: str = "0.0.0.0",
    namespace: str = "",
):
    """The single-robot path: own a server on `port`, put one arm on it, start it.

    A thin wrapper over `attach_ros`, kept so callers that only ever want one robot need
    no knowledge of fleets. `spawn_robot.py` builds a `RobotFleet` directly, because it
    may be asked for several robots at once and they must share the one port.
    """
    from ros_surfaces import RobotFleet

    fleet = RobotFleet(port=port, host=host)
    fleet.attach(namespace, attach_ros, view=view, model=model, task=task,
                 cameras=cameras, jpeg_quality=jpeg_quality, control_hz=control_hz,
                 scene_option=scene_option)
    fleet.start()
    print(f"SO-101 on ws://{host}:{port} under namespace {namespace or '<bare>'}",
          file=sys.stderr)
    return fleet
