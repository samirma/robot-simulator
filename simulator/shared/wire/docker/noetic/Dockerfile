# Wire base image, ROS 1: ROS Noetic (the distribution every ROS 1 robot of the robot
# specification records) with rosbridge_suite and the stock ROS packages the recorded
# boots run. Built by spawn.sh on first use. Everything is pinned: the base image by
# digest (multi-arch; arm64 on Apple silicon) and every package by its exact version from
# the final Noetic snapshot the base image's apt source points at.
FROM ros:noetic-ros-base@sha256:72b8bc59035dc0a5b8e07aae28c16caa84192971d72d207c72ed734fb1d5e97d
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-rosbridge-server=0.11.17-1focal.20250520.085849 \
      ros-noetic-rosbridge-library=0.11.17-1focal.20250520.* \
      ros-noetic-rosapi=0.11.17-1focal.20250520.* \
      ros-noetic-robot-state-publisher=1.15.3-1focal.20250520.020302 \
      ros-noetic-joint-state-publisher=1.15.2-1focal.20250520.012744 \
      ros-noetic-tf=1.13.4-1focal.20250520.015357 \
      ros-noetic-tf2-ros ros-noetic-xacro ros-noetic-sensor-msgs ros-noetic-nav-msgs \
      ros-noetic-trajectory-msgs ros-noetic-control-msgs ros-noetic-std-srvs \
      ros-noetic-actionlib ros-noetic-actionlib-msgs ros-noetic-diagnostic-msgs \
      ros-noetic-image-transport ros-noetic-camera-info-manager \
      ros-noetic-cv-bridge \
      python3-yaml python3-numpy python3-catkin-tools git build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO=noetic RSIM_DIALECT=ros1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=noetic org.rsim.rosbridge=0.11.17
