# Wire base image, ROS 2: ROS Humble (the myCobot 280 boot's distribution) with
# rosbridge_suite and the stock ROS 2 packages its boot runs. Built by spawn.sh on first use,
# from pinned sources: the base image by digest (multi-arch; arm64 on Apple silicon) and
# every ROS package from the dated snapshot of the ROS 2 repository below (snapshots.ros.org,
# signed by the ROS snapshot key, pinned by fingerprint), never the live repository the base
# image points at; the ROS packages the base image already holds are brought to that same
# snapshot. The stock packages the boot runs are also pinned by the release the interface
# file records (robot_state_publisher 3.0.3, joint_state_publisher 2.4.0, and rclcpp 16.0.21,
# the release of its recorded revision cfdb3b7; the build suffix is per architecture, hence
# the `*`). Ubuntu's own packages come from the base image's jammy archive.
FROM ros:humble-ros-base@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN key=4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA && export GNUPGHOME="$(mktemp -d)" \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x$key" \
         | gpg --batch --import \
    && gpg --batch --export "$key" > /usr/share/keyrings/ros-snapshots-archive-keyring.gpg \
    && test -s /usr/share/keyrings/ros-snapshots-archive-keyring.gpg \
    && gpgconf --kill all && rm -rf "$GNUPGHOME" \
    && dpkg --purge ros2-apt-source \
    && echo "deb [ signed-by=/usr/share/keyrings/ros-snapshots-archive-keyring.gpg ] http://snapshots.ros.org/humble/2026-09-14/ubuntu jammy main" \
         > /etc/apt/sources.list.d/ros2-snapshots.list \
    && ! grep -rqs packages.ros.org /etc/apt/sources.list /etc/apt/sources.list.d
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade $(dpkg-query -W \
         -f='${db:Status-Abbrev} ${Package}\n' 'ros-humble-*' | awk '$1 == "ii" {print $2}') \
    && apt-get install -y --no-install-recommends \
      ros-humble-rosbridge-server=2.0.8-1jammy.* \
      ros-humble-rosbridge-library=2.0.8-1jammy.* \
      ros-humble-rosapi=2.0.8-1jammy.* \
      ros-humble-rosapi-msgs=2.0.8-1jammy.* \
      ros-humble-robot-state-publisher=3.0.3-2jammy.* \
      ros-humble-joint-state-publisher=2.4.0-1jammy.* \
      ros-humble-rclcpp=16.0.21-1jammy.* \
      ros-humble-xacro=2.1.1-1jammy.* \
      ros-humble-tf2-ros ros-humble-sensor-msgs ros-humble-trajectory-msgs \
      ros-humble-control-msgs ros-humble-std-srvs ros-humble-controller-manager-msgs \
      python3-yaml python3-numpy git build-essential python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
ENV ROS_DISTRO=humble RSIM_DIALECT=ros2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=humble org.rsim.rosbridge=2.0.8 org.rsim.ros_snapshot=humble/2026-09-14
