# Wire base image, ROS 2: ROS Humble (the distribution every ROS 2 robot of the robot
# specification records) with rosbridge_suite and the stock ROS 2 packages the recorded
# boots run. Built by spawn.sh on first use. The base image is pinned by digest
# (multi-arch; arm64 on Apple silicon) and rosbridge_suite by its exact release.
FROM ros:humble-ros-base@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-rosbridge-server=2.0.8-1jammy.* \
      ros-humble-rosbridge-library=2.0.8-1jammy.* \
      ros-humble-rosapi=2.0.8-1jammy.* \
      ros-humble-rosapi-msgs=2.0.8-1jammy.* \
      ros-humble-robot-state-publisher ros-humble-joint-state-publisher \
      ros-humble-xacro ros-humble-tf2-ros ros-humble-sensor-msgs ros-humble-trajectory-msgs \
      ros-humble-control-msgs ros-humble-std-srvs ros-humble-controller-manager-msgs \
      python3-yaml python3-numpy git build-essential python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO=humble RSIM_DIALECT=ros2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=humble org.rsim.rosbridge=2.0.8
