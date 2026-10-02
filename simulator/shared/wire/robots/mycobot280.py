"""myCobot 280 Pi + adaptive gripper (ROS 2 Humble): the wire of mycobot_280pi's
`slider_control_adaptive_gripper.launch.py` (gui:=false).

Stock: /robot_state_publisher with the boot's robot_description (`xacro` of the URDF) and
parameters. Simulated: /slider_control_adaptive_gripper, the node that forwards every
/joint_states message to the arm (pymycobot `send_angles(angles, 25)` and
`set_gripper_value(v, 80)`), here to the arm's position servos in the simulation.
"""

from __future__ import annotations

import math
import os
import subprocess
import threading
import time

import common
from robots import _plan

#: pymycobot speed 25 (of 1..100) and gripper speed 80 are not documented in deg/s
#: (robots_specs/mycobot280/import.md: "the simulator must pace motions"). Estimate: the
#: 280 series' top joint speed of about 160 deg/s scaled by the speed percentage, i.e.
#: 40 deg/s for the arm; the gripper crosses its range in about 0.5 s at speed 80.
ARM_SPEED_RAD_S = math.radians(160.0) * 0.25
GRIPPER_SPEED_RAD_S = 0.89 / 0.5
CONTROL_HZ = 100.0

ARM = ["joint2_to_joint1", "joint3_to_joint2", "joint4_to_joint3", "joint5_to_joint4",
       "joint6_to_joint5", "joint6output_to_joint6"]
GRIPPER = "gripper_controller"


def _xacro(robot) -> str:
    urdf = robot.path(robot.urdf)
    distro = os.environ.get("ROS_DISTRO", "humble")
    out = subprocess.run(["bash", "-c", f"source /opt/ros/{distro}/setup.bash && xacro {urdf}"],
                         stdout=subprocess.PIPE, check=True, text=True)
    return out.stdout


def rsp_params_file(robot, iface, node="/robot_state_publisher", overrides=None) -> str:
    """A ROS 2 parameter file with the node's recorded parameters."""
    import yaml

    params = {}
    for row in common.params_of(iface, node):
        v = (overrides or {}).get(row["name"], common.param_value(robot, row))
        if isinstance(v, dict) and "generated" in v:
            raise SystemExit(f"{node} {row['name']} is generated and has no value")
        cur = params
        cur[row["name"]] = v   # dotted names are flat keys in a ROS 2 parameter file
    path = f"/tmp/{robot.id}_{node.strip('/').replace('/', '_')}.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump({node: {"ros__parameters": params}}, fh)
    return path


def plan(robot, iface, describe):
    p = _plan.Plan()
    desc = _xacro(robot)
    f = rsp_params_file(robot, iface, overrides={"robot_description": desc})
    p.add("/robot_state_publisher", ["ros2", "run", "robot_state_publisher",
                                     "robot_state_publisher", "--ros-args", "-r",
                                     "__node:=robot_state_publisher", "--params-file", f])
    _plan.ros2_emulated(p, ["/slider_control_adaptive_gripper"])
    return p


class Slider:
    """/slider_control_adaptive_gripper: each /joint_states message is one command.

    position[0..5] are the six arm angles (rad; the node sends them in degrees rounded to
    0.01 with send_angles at speed 25); position[6] is the gripper, mapped to the pymycobot
    value int((p + 0.74) / 0.89 * 100) and sent with set_gripper_value at speed 80 (a
    message with exactly six positions sends 0, closing it). As on the robot, a message
    with fewer than six positions, an angle outside the limits or a gripper value outside
    0..100 raises inside the callback and ends the node. The servos hold the last target:
    there is no watchdog."""

    def __init__(self, node):
        self.node = node
        arm_row = next(m for m in node.iface["motions"] if m["id"] == "arm")
        self.limits = []
        for f in arm_row["command"]["fields"]:
            if f["field"].startswith("position["):
                self.limits.append((float(f["min"]), float(f["max"])))
        self.target = None
        self.setpoint = None
        self.lock = threading.Lock()

    def start(self):
        link = self.node.link
        acts = link.actuators()
        missing = [a for a in ARM + [GRIPPER] if a not in acts]
        if missing:
            raise SystemExit(f"model lacks actuators {missing}")
        state = {}
        ev = threading.Event()

        def first(h, _):
            if not ev.is_set():
                state.update(h["joints"])
                ev.set()
        link.subscribe("state", 20.0, first, joints=True)
        ev.wait(10)
        with self.lock:
            self.setpoint = [state[j][0] for j in ARM] + [state[GRIPPER][0]]
            self.target = list(self.setpoint)
        threading.Thread(target=self._loop, daemon=True).start()

    def on_message(self, topic, msg):
        pos = list(msg.position)
        if len(pos) < 6:
            self._die(f"send_angles got {len(pos)} angles")
        deg = [round(math.degrees(p), 2) for p in pos[:6]]
        for i, (d, (lo, hi)) in enumerate(zip(deg, self.limits)):
            if not math.degrees(lo) - 1e-6 <= d <= math.degrees(hi) + 1e-6:
                self._die(f"angle {d} of joint {i + 1} is outside its limits")
        value = int((pos[6] + 0.74) / 0.89 * 100) if len(pos) > 6 else 0
        if not 0 <= value <= 100:
            self._die(f"gripper value {value} is outside 0..100")
        with self.lock:
            self.target = [math.radians(d) for d in deg] + [value / 100.0 * 0.89 - 0.74]

    def _die(self, why):
        print(f"[slider_control_adaptive_gripper] pymycobot raised: {why}; the node ends",
              flush=True)
        os._exit(1)

    def _loop(self):
        dt = 1.0 / CONTROL_HZ
        while True:
            t0 = time.monotonic()
            with self.lock:
                tgt, sp = self.target, self.setpoint
                # send_angles moves every joint together: the joint with the farthest to
                # go sets the pace and the others are scaled to arrive with it.
                arm_err = [t - s for t, s in zip(tgt[:6], sp[:6])]
                far = max((abs(e) for e in arm_err), default=0.0)
                step = ARM_SPEED_RAD_S * dt
                k = 1.0 if far <= step else step / far
                for i in range(6):
                    sp[i] += arm_err[i] * k
                g = tgt[6] - sp[6]
                gs = GRIPPER_SPEED_RAD_S * dt
                sp[6] += max(-gs, min(gs, g))
                cmd = dict(zip(ARM + [GRIPPER], sp))
            self.node.link.ctrl(cmd)
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))


BEHAVIOURS = {"/slider_control_adaptive_gripper": Slider}
