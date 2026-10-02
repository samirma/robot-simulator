# Wire base image, ROS 2 Jazzy (the SO-101 boot's distribution) with rosbridge_suite, and the
# stock packages its boot runs: ros2_control's controller_manager and the ros2_controllers the
# bringup spawns, robot_state_publisher and xacro. Built by spawn.sh on first use. The base
# image is pinned by digest (multi-arch; arm64 on Apple silicon), the ros2_controllers
# release by its version (4.42.1, as the interface file records); parallel_gripper_controller
# is rebuilt from that release's pinned revision with one uninitialised member fixed (below).
FROM ros:jazzy-ros-base@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-rosbridge-server ros-jazzy-rosbridge-library ros-jazzy-rosapi ros-jazzy-rosapi-msgs \
      ros-jazzy-controller-manager ros-jazzy-hardware-interface ros-jazzy-ros2-control \
      ros-jazzy-joint-trajectory-controller=4.42.1-1noble.* \
      ros-jazzy-joint-state-broadcaster=4.42.1-1noble.* \
      ros-jazzy-parallel-gripper-controller=4.42.1-1noble.* \
      ros-jazzy-position-controllers=4.42.1-1noble.* \
      ros-jazzy-robot-state-publisher ros-jazzy-xacro ros-jazzy-tf2-ros \
      ros-jazzy-sensor-msgs ros-jazzy-std-srvs ros-jazzy-control-msgs ros-jazzy-pal-statistics-msgs \
      python3-yaml python3-numpy git build-essential cmake python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
# parallel_gripper_controller 4.42.1 (ros2_controllers aacd842, the recorded revision) copies its
# member `computed_command_` into the action result's state.effort[0] but never initialises or
# writes it, so /gripper_controller/gripper_cmd returns whatever the heap held (e.g. 5.2e+170).
# The recorded boot configures no effort state interface (state_interfaces: position, velocity;
# /joint_states carries no effort): effort is not measured, so the wire reports 0.0. Rebuilt from
# the pinned revision with that one member zero-initialised, in place of the apt library; nothing
# else changes.
ARG ROS2_CONTROLLERS_REV=aacd842600a09d556b983ac3d53a0983e9ebcbb1
RUN apt-get update && apt-get install -y --no-install-recommends ros-jazzy-ros2-control-cmake \
    && rm -rf /var/lib/apt/lists/* \
    && git init -q /tmp/rc && cd /tmp/rc \
    && git fetch -q --depth 1 https://github.com/ros-controls/ros2_controllers "$ROS2_CONTROLLERS_REV" \
    && git checkout -q FETCH_HEAD \
    && hdr=parallel_gripper_controller/include/parallel_gripper_controller/parallel_gripper_action_controller.hpp \
    && grep -q '^  double computed_command_;' "$hdr" \
    && sed -i 's/^  double computed_command_;/  double computed_command_ = 0.0;/' "$hdr" \
    && grep -q '^  double computed_command_ = 0.0;' "$hdr" \
    && source /opt/ros/jazzy/setup.bash \
    && cmake -S parallel_gripper_controller -B /tmp/pgc -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/opt/ros/jazzy -DBUILD_TESTING=OFF \
    && cmake --build /tmp/pgc -j"$(nproc)" \
    && cmake --install /tmp/pgc \
    && cd / && rm -rf /tmp/rc /tmp/pgc
ENV ROS_DISTRO=jazzy RSIM_DIALECT=ros2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=jazzy
