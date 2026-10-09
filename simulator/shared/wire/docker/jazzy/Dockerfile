# Wire base image, ROS 2 Jazzy (the reBot Arm B601-RS boot's distribution) with rosbridge_suite
# and the stock packages its boot runs: robot_state_publisher with tf2_ros, and the client
# libraries and launch the vendor nodes run on; xacro and colcon for robot layers. Built by
# spawn.sh on first use, from pinned sources: the base image by digest (multi-arch; arm64 on
# Apple silicon) and every ROS package from the dated snapshot of the ROS 2 repository below
# (snapshots.ros.org, signed by the ROS snapshot key, pinned by fingerprint), never the live
# repository the base image points at; the ROS packages the base image already holds are
# brought to that same snapshot. The stock packages the boot runs are also pinned by the
# release the interface file records (robot_state_publisher 3.3.4, rclcpp 28.1.22, rclpy
# 7.1.12; the build suffix is per architecture, hence the `*`). The recorded releases no
# snapshot carries (geometry2 0.36.23, rmw 7.3.4 and launch 3.4.12: the snapshot has 0.36.22,
# 7.3.3 and 3.4.11) are built from their recorded revisions in their place. Every Ubuntu
# package comes from the dated snapshot of the Ubuntu archive below (snapshot.ubuntu.com: the
# noble archive, every pocket the base image uses, as published at 2026-10-08 21:30 UTC, the
# state the wire images were built and checked against; signed by the Ubuntu archive key the
# base image holds; one URL serves every architecture), never the live archive the base image
# points at. Robot layers built on this image install from these same snapshots.
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
    && test -s /usr/share/keyrings/ubuntu-archive-keyring.gpg \
    && printf '%s\n' 'Types: deb' 'URIs: https://snapshot.ubuntu.com/ubuntu/20261008T213000Z' \
         'Suites: noble noble-updates noble-backports noble-security' \
         'Components: main universe restricted multiverse' \
         'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
         > /etc/apt/sources.list.d/ubuntu.sources \
    && ! grep -rqs packages.ros.org /etc/apt/sources.list /etc/apt/sources.list.d \
    && ! grep -rqsE '(archive|ports|security)\.ubuntu\.com' /etc/apt/sources.list /etc/apt/sources.list.d
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade $(dpkg-query -W \
         -f='${db:Status-Abbrev} ${Package}\n' 'ros-jazzy-*' | awk '$1 == "ii" {print $2}') \
    && apt-get install -y --no-install-recommends \
      ros-jazzy-rosbridge-server=2.7.1-1noble.* \
      ros-jazzy-rosbridge-library=2.7.1-1noble.* \
      ros-jazzy-rosapi=2.7.1-1noble.* \
      ros-jazzy-rosapi-msgs=2.7.1-1noble.* \
      ros-jazzy-robot-state-publisher=3.3.4-1noble.* \
      ros-jazzy-rclcpp=28.1.22-1noble.* \
      ros-jazzy-rclpy=7.1.12-1noble.* \
      ros-jazzy-xacro=2.1.1-1noble.* \
      ros-jazzy-tf2-ros ros-jazzy-sensor-msgs ros-jazzy-std-srvs ros-jazzy-control-msgs \
      ros-jazzy-ament-cmake-ros \
      python3-yaml python3-numpy git build-essential cmake python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
# geometry2 0.36.23 (f702874) and rmw 7.3.4 (8cb94a8), the releases the interface file records
# for robot_state_publisher's tf2_ros broadcasters and the rmw QoS profiles, built from those
# revisions in place of the snapshot's 0.36.22 and 7.3.3: tf2 with tf2_ros, and rmw.
ARG GEOMETRY2_REV=f702874b1c8535d6a038230ab2cda0ba5d521ebd
ARG RMW_REV=8cb94a87cf3f4049abe2d563af893517bf336930
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
    && git init -q /tmp/rmw \
    && git -C /tmp/rmw fetch -q --depth 1 https://github.com/ros2/rmw "$RMW_REV" \
    && git -C /tmp/rmw checkout -q FETCH_HEAD \
    && build /tmp/rmw/rmw 7.3.4 \
    && rm -rf /tmp/g2 /tmp/rmw /tmp/build
# launch 3.4.12 (b63c1b8), the release the interface file records for the boot's Command
# substitution, installed from that revision in place of the snapshot's 3.4.11 as its
# ament_python build installs it: the Python package, its data files and its metadata.
ARG LAUNCH_REV=b63c1b82687038dad9a5d914c631f6681a550abe
RUN git init -q /tmp/launch \
    && git -C /tmp/launch fetch -q --depth 1 https://github.com/ros2/launch "$LAUNCH_REV" \
    && git -C /tmp/launch checkout -q FETCH_HEAD \
    && site=/opt/ros/jazzy/lib/python3.12/site-packages \
    && rm -rf "$site/launch" "$site"/launch-*.egg-info \
    && cd /tmp/launch/launch \
    && python3 setup.py -q install --prefix /opt/ros/jazzy --install-lib "$site" \
         --install-data /opt/ros/jazzy --single-version-externally-managed \
         --record /tmp/launch.files \
    && ! grep -v -e "^$site/" -e "^/opt/ros/jazzy/share/" /tmp/launch.files \
    && grep -q "<version>3.4.12</version>" /opt/ros/jazzy/share/launch/package.xml \
    && python3 -c "import sys; sys.path.insert(0, '$site'); import launch, importlib.metadata as m; assert m.version('launch') == '3.4.12'" \
    && cd / && rm -rf /tmp/launch /tmp/launch.files
ENV ROS_DISTRO=jazzy RSIM_DIALECT=ros2 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
LABEL org.rsim.wire=jazzy org.rsim.ros_snapshot=jazzy/2026-09-11 \
      org.rsim.ubuntu_snapshot=noble/20261008T213000Z
