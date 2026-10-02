# ROSMASTER X3 PLUS wire image: the Noetic wire base plus what its recorded boot runs that
# needs no hardware (imu_filter_madgwick, robot_localization's EKF, joy) and:
#   * the vendor's yahboomcar_msgs, yahboomcar_bringup (the real driver_node script,
#     base_node and dynamic_reconfigure configs) and yahboomcar_ctrl (yahboom_joy), from
#     the pinned workspace archive yahboomcar_ws.zip (sha256 13d752e0..., put into the build
#     context by spawn.sh from the verified cache). The driver runs unchanged against the
#     simulator's board library (wire/robots/sim_libs/Rosmaster_Lib.py);
#   * orbbec_camera's messages and services. The vendor's own copy (software.zip in the
#     same Drive archive) could not be fetched (Drive quota); the definitions come from
#     Orbbec's public OrbbecSDK_ROS1 v1.4.2 (commit 7ef885a4), the same v1 service set the
#     interface file records -- an adaptation documented in simulator/README.md.
ARG BASE
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-imu-filter-madgwick ros-noetic-robot-localization ros-noetic-joy \
      ros-noetic-dynamic-reconfigure ros-noetic-message-generation \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/yahboomcar_msgs /opt/rsim_ws/src/yahboomcar_msgs
COPY vendor/yahboomcar_bringup /opt/rsim_ws/src/yahboomcar_bringup
COPY vendor/yahboomcar_ctrl /opt/rsim_ws/src/yahboomcar_ctrl
COPY orbbec_camera_msgs /opt/rsim_ws/src/orbbec_camera
RUN mkdir -p /opt/rsim_src && cd /opt/rsim_src && git init -q ob && cd ob \
 && git remote add origin https://github.com/orbbec/OrbbecSDK_ROS1.git \
 && git fetch -q --depth 1 origin 7ef885a4a168df02948ca97da216751a83d40471 \
 && git checkout -q FETCH_HEAD \
 && cp -r msg srv /opt/rsim_ws/src/orbbec_camera/ \
 && source /opt/ros/noetic/setup.bash && cd /opt/rsim_ws \
 && catkin_make -DCMAKE_BUILD_TYPE=Release > /tmp/build.log 2>&1 || (tail -40 /tmp/build.log; exit 1)
LABEL org.rsim.wire=rosmaster_x3_plus
