# Wire base image, ROS 1: ROS Noetic (the distribution every ROS 1 robot of the robot
# specification records) with rosbridge_suite and the stock ROS packages the recorded
# boots run. Built by spawn.sh on first use, from pinned sources: the base image by digest
# (multi-arch; arm64 on Apple silicon) and every ROS package from the dated Noetic snapshot
# below (snapshots.ros.org; the last Noetic sync, identical to the `final` snapshot the base
# image points at), the boots' stock nodes also by their exact release (the build suffix is
# per architecture, hence the `*`). Ubuntu's own packages come from the base image's focal
# archive.
FROM ros:noetic-ros-base@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN test -s /usr/share/keyrings/ros1-snapshots-archive-keyring.gpg \
    && echo "deb [ signed-by=/usr/share/keyrings/ros1-snapshots-archive-keyring.gpg ] http://snapshots.ros.org/noetic/2025-05-29/ubuntu focal main" \
         > /etc/apt/sources.list.d/ros1-snapshots.list \
    && ! grep -rqs 'packages.ros.org\|noetic/final' /etc/apt/sources.list /etc/apt/sources.list.d
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-rosbridge-server=0.11.17-1focal.* \
      ros-noetic-rosbridge-library=0.11.17-1focal.* \
      ros-noetic-rosapi=0.11.17-1focal.* \
      ros-noetic-robot-state-publisher=1.15.3-1focal.* \
      ros-noetic-joint-state-publisher=1.15.2-1focal.* \
      ros-noetic-tf=1.13.4-1focal.* \
      ros-noetic-tf2-ros ros-noetic-xacro ros-noetic-sensor-msgs ros-noetic-nav-msgs \
      ros-noetic-trajectory-msgs ros-noetic-control-msgs ros-noetic-std-srvs \
      ros-noetic-actionlib ros-noetic-actionlib-msgs ros-noetic-diagnostic-msgs \
      ros-noetic-image-transport ros-noetic-camera-info-manager \
      ros-noetic-cv-bridge \
      python3-yaml python3-numpy python3-catkin-tools git build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO=noetic RSIM_DIALECT=ros1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=noetic org.rsim.rosbridge=0.11.17 org.rsim.ros_snapshot=noetic/2025-05-29
