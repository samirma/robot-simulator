"""myAGV (ROS 1 Noetic): the wire of `myagv_active.launch` plus the approved usb_cam boot.

Stock nodes, as the boot runs them: joint_state_publisher, robot_state_publisher, the
three tf static_transform_publisher mounts and the vendor's own (modified) robot_pose_ekf,
built from the pinned myagv_ros source. Simulated drivers: /myagv_odometry_node (the
base MCU: mecanum wheels driven through the simulation, odometry and IMU as the vendor
driver reports them), /ydlidar_lidar_publisher (the X2L) and /usb_cam (the CSI camera).
"""

from __future__ import annotations

import math
import threading
import time

import common
from robots import _plan
from robots._ros1_drivers import UsbCam, YdLidar


def plan(robot, iface, describe):
    p = _plan.Plan()
    p.add("/joint_state_publisher", ["rosrun", "joint_state_publisher", "joint_state_publisher",
                                     *_plan.ros1_name("/joint_state_publisher")])
    p.add("/robot_state_publisher", ["rosrun", "robot_state_publisher", "robot_state_publisher",
                                     *_plan.ros1_name("/robot_state_publisher")])
    _plan.ros1_static_tf(p, iface)
    # The vendor's robot_pose_ekf, remapped as myagv_active.launch does (imu_data -> /imu).
    p.add("/robot_pose_ekf", ["rosrun", "robot_pose_ekf", "robot_pose_ekf",
                              *_plan.ros1_name("/robot_pose_ekf"), "imu_data:=/imu"])
    for node in ("/myagv_odometry_node", "/ydlidar_lidar_publisher", "/usb_cam"):
        _plan.ros1_emulated(p, node)
    return p


def drive_row(iface):
    return next(m for m in iface["motions"] if m["id"] == "drive")


class Odometry:
    """/myagv_odometry_node: myAGV.cpp's behaviour against a simulated MCU.

    The last /cmd_vel is held (no watchdog) and clamped to +-1 and truncated to the MCU's
    0.01 resolution; the MCU's mecanum mixing drives the four wheel velocity servos. Each
    MCU report (the simulation's state stream, 100 Hz) gives the chassis velocity from the
    wheel speeds at 0.01 resolution, the IMU and the yaw in degrees; /odom integrates the
    velocity with the accumulated yaw (changes under 0.1 deg per report dropped, as the
    driver does), and /imu, /Voltage and /voltage_backup follow each report."""

    publishes = ("/odom", "/imu", "/Voltage", "/voltage_backup")

    def __init__(self, node):
        self.node = node
        self.kin = drive_row(node.iface)["kinematics"]
        self.wheels = list(self.kin["wheel_joints"])
        self.cmd = (0.0, 0.0, 0.0)
        self.x = self.y = 0.0
        self.last_theta = None
        self.acc_theta = 0.0
        self.yaw0 = None
        self.last_stamp = None
        self.lock = threading.Lock()
        rate = common.rate_of(common.topic_row(node.iface, "/odom")) or 100.0
        self.rate = rate

    def start(self):
        self.node.link.subscribe("state", self.rate, self._report, joints=True, base=True,
                                 imu_sites=["imu_link"])
        self._send()

    @staticmethod
    def _mcu(v):
        v = max(-1.0, min(1.0, v))
        return int(v * 100) / 100.0      # static_cast<signed char>(v * 100)

    def on_message(self, topic, msg):
        if topic == "/cmd_vel":
            with self.lock:
                self.cmd = (self._mcu(msg.linear.x), self._mcu(msg.linear.y),
                            self._mcu(msg.angular.z))
            self._send()

    def _send(self):
        vx, vy, wz = self.cmd
        w = _plan.mecanum_ik(self.kin, vx, vy, wz)
        self.node.link.ctrl(dict(zip(self.wheels, w)))

    def _report(self, h, _payload):
        stamp = h["stamp"]
        joints = h["joints"]
        w = [joints[j][1] for j in self.wheels]
        vx, vy, wz = _plan.mecanum_fk(self.kin, w)
        q = lambda v: max(-1.28, min(1.27, round(v * 100) / 100.0))
        vx, vy, wz = q(vx), q(vy), q(wz)
        base = h["base"]
        yaw_world = math.degrees(common.yaw_of(base["quat"]))
        if self.yaw0 is None:
            self.yaw0 = yaw_world
        yaw = _plan.wrap_deg(yaw_world - self.yaw0)       # MCU yaw, relative to power-on
        if self.last_theta is None:
            self.last_theta = yaw
        d = yaw - self.last_theta
        if -0.1 < d < 0.1:
            d = 0.0
        self.acc_theta += d
        self.last_theta = yaw
        dt = 0.0 if self.last_stamp is None else stamp - self.last_stamp
        self.last_stamp = stamp
        theta = math.radians(self.acc_theta)
        self.x += (vx * math.cos(theta) - vy * math.sin(theta)) * dt
        self.y += (vx * math.sin(theta) + vy * math.cos(theta)) * dt
        n = self.node
        odom = n.new("/odom", stamp=stamp)
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        qz, qw = math.sin(theta / 2), math.cos(theta / 2)
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = qz, qw
        cov = [0.0] * 36
        for i, v in ((0, 1e-9), (7, 1e-3), (8, 1e-9), (14, 1e6), (21, 1e6), (28, 1e6), (35, 1e-9)):
            cov[i] = v
        odom.pose.covariance = cov
        odom.twist.covariance = cov
        odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx, vy
        odom.twist.twist.angular.z = wz
        n.publish("/odom", odom)
        imu = n.new("/imu", stamp=stamp)
        imu.header.frame_id = "imu_link"
        yr = math.radians(yaw)
        imu.orientation.z, imu.orientation.w = math.sin(yr / 2), math.cos(yr / 2)
        s = h["imu"]["imu_link"]
        # The MCU reports raw units the driver does not convert (estimate: deg/s, m/s^2),
        # at 0.1 and 0.001 resolution.
        gx, gy, gz = (round(math.degrees(v), 1) for v in s["gyro"])
        ax, ay, az = (round(v, 3) for v in s["accel"])
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = gx, gy, gz
        imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z = ax, ay, az
        oc = [0.0] * 9
        oc[0], oc[4], oc[8] = 1e6, 1e6, 1e-6
        imu.orientation_covariance = oc
        imu.angular_velocity_covariance = list(oc)
        n.publish("/imu", imu)
        v = n.new("/Voltage")
        v.data = 12.0   # estimate: nominal 12 V pack (the battery level is not simulated)
        n.publish("/Voltage", v)
        vb = n.new("/voltage_backup")
        vb.data = 0.0   # no backup battery fitted
        n.publish("/voltage_backup", vb)


class Camera(UsbCam):
    image_topic = "/usb_cam/image_raw"


BEHAVIOURS = {
    "/myagv_odometry_node": Odometry,
    "/ydlidar_lidar_publisher": YdLidar,
    "/usb_cam": Camera,
}
