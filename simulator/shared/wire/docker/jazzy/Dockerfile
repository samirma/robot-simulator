# Wire base image, ROS 2 Jazzy (the SO-101 boot's distribution) with rosbridge_suite, and the
# stock packages its boot runs: ros2_control's controller_manager and the ros2_controllers the
# bringup spawns, robot_state_publisher and xacro. Built by spawn.sh on first use, from pinned
# sources: the base image by digest (multi-arch; arm64 on Apple silicon) and every ROS package
# from the dated snapshot of the ROS 2 repository below (snapshots.ros.org, signed by the ROS
# snapshot key, pinned by fingerprint), never the live repository the base image points at;
# the ROS packages the base image already holds are brought to that same snapshot. The stock
# packages the boot runs are also pinned by the release the interface file records
# (ros2_controllers 4.42.1, robot_state_publisher 3.3.4, diagnostics 4.2.7, pal_statistics
# 2.8.2; the build suffix is per architecture, hence the `*`). The two recorded releases no
# snapshot carries (ros2_control 4.48.1 and geometry2 0.36.23: the snapshot has 4.48.0 and
# 0.36.22) are built from their recorded revisions, and parallel_gripper_controller is rebuilt
# from its recorded revision with one uninitialised member fixed (below). Ubuntu's own
# packages come from the base image's noble archive.
FROM ros:jazzy-ros-base@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN key=4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA && export GNUPGHOME="$(mktemp -d)" \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x$key" \
         | gpg --batch --import \
    && gpg --batch --export "$key" > /usr/share/keyrings/ros-snapshots-archive-keyring.gpg \
    && test -s /usr/share/keyrings/ros-snapshots-archive-keyring.gpg \
    && gpgconf --kill all && rm -rf "$GNUPGHOME" \
    && dpkg --purge ros2-apt-source \
    && echo "deb [ signed-by=/usr/share/keyrings/ros-snapshots-archive-keyring.gpg ] http://snapshots.ros.org/jazzy/2026-09-11/ubuntu noble main" \
         > /etc/apt/sources.list.d/ros2-snapshots.list \
    && ! grep -rqs packages.ros.org /etc/apt/sources.list /etc/apt/sources.list.d
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade $(dpkg-query -W \
         -f='${db:Status-Abbrev} ${Package}\n' 'ros-jazzy-*' | awk '$1 == "ii" {print $2}') \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-rosbridge-server=2.7.1-1noble.* \
      ros-jazzy-rosbridge-library=2.7.1-1noble.* \
      ros-jazzy-rosapi=2.7.1-1noble.* \
      ros-jazzy-rosapi-msgs=2.7.1-1noble.* \
      ros-jazzy-controller-manager ros-jazzy-hardware-interface ros-jazzy-controller-interface \
      ros-jazzy-joint-trajectory-controller=4.42.1-1noble.* \
      ros-jazzy-joint-state-broadcaster=4.42.1-1noble.* \
      ros-jazzy-parallel-gripper-controller=4.42.1-1noble.* \
      ros-jazzy-position-controllers=4.42.1-1noble.* \
      ros-jazzy-robot-state-publisher=3.3.4-1noble.* \
      ros-jazzy-diagnostic-updater=4.2.7-1noble.* \
      ros-jazzy-pal-statistics=2.8.2-1noble.* \
      ros-jazzy-pal-statistics-msgs=2.8.2-1noble.* \
      ros-jazzy-xacro=2.1.1-1noble.* \
      ros-jazzy-tf2-ros ros-jazzy-sensor-msgs ros-jazzy-std-srvs ros-jazzy-control-msgs \
      ros-jazzy-ros2-control-cmake ros-jazzy-ament-cmake-gen-version-h ros-jazzy-ament-cmake-ros \
      python3-yaml python3-numpy git build-essential cmake python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
# ros2_control 4.48.1 (b7f1a5d) and geometry2 0.36.23 (f702874), the releases the interface
# file records for controller_manager / ros2_control_node and for robot_state_publisher's tf2_ros
# broadcasters, built from those revisions in place of the snapshot's 4.48.0 and 0.36.22: every
# ros2_control package the boot loads, and tf2 with tf2_ros. Two compile jobs keep the large
# translation units within a default Docker VM's memory.
ARG ROS2_CONTROL_REV=b7f1a5d420443c7a079f4c81e80d5e39759acf4c
ARG GEOMETRY2_REV=f702874b1c8535d6a038230ab2cda0ba5d521ebd
RUN source /opt/ros/jazzy/setup.bash \
    && build() { cmake -S "$1" -B "/tmp/build/${1##*/}" -DCMAKE_BUILD_TYPE=Release \
                   -DCMAKE_INSTALL_PREFIX=/opt/ros/jazzy -DBUILD_TESTING=OFF \
                 && cmake --build "/tmp/build/${1##*/}" -j2 \
                 && cmake --install "/tmp/build/${1##*/}" \
                 && grep -q "<version>$2</version>" "/opt/ros/jazzy/share/${1##*/}/package.xml"; } \
    && git init -q /tmp/g2 \
    && git -C /tmp/g2 fetch -q --depth 1 https://github.com/ros2/geometry2 "$GEOMETRY2_REV" \
    && git -C /tmp/g2 checkout -q FETCH_HEAD \
    && for p in tf2 tf2_ros; do build "/tmp/g2/$p" 0.36.23 || exit 1; done \
    && git init -q /tmp/rc2 \
    && git -C /tmp/rc2 fetch -q --depth 1 https://github.com/ros-controls/ros2_control "$ROS2_CONTROL_REV" \
    && git -C /tmp/rc2 checkout -q FETCH_HEAD \
    && for p in controller_manager_msgs joint_limits hardware_interface controller_interface \
                controller_manager; do build "/tmp/rc2/$p" 4.48.1 || exit 1; done \
    && rm -rf /tmp/g2 /tmp/rc2 /tmp/build
# parallel_gripper_controller 4.42.1 (ros2_controllers aacd842, the recorded revision) copies its
# member `computed_command_` into the action result's state.effort[0] but never initialises or
# writes it, so /gripper_controller/gripper_cmd returns whatever the heap held (e.g. 5.2e+170).
# The recorded boot configures no effort state interface (state_interfaces: position, velocity;
# /joint_states carries no effort): effort is not measured, so the wire reports 0.0. Rebuilt from
# the pinned revision with that one member zero-initialised, in place of the apt library; nothing
# else changes.
ARG ROS2_CONTROLLERS_REV=aacd842600a09d556b983ac3d53a0983e9ebcbb1
RUN git init -q /tmp/rc && cd /tmp/rc \
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
LABEL org.rsim.wire=jazzy org.rsim.ros_snapshot=jazzy/2026-09-11 org.rsim.ros2_control=4.48.1
