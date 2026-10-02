# AiNex wire image: the Noetic wire base plus the vendor's own workspace at the path its
# boot uses (/home/ubuntu/ros_ws), from the pinned Hiwonder/ainex revision: the message
# packages (ainex_interfaces, ros_robot_controller, sensor_msgs_ext), the board node
# (ros_robot_controller), the gait engine and controller (ainex_kinematics, with its
# aarch64 CPython 3.8 extension modules walking_module.so and kinematics.so), ainex_sdk,
# imu_calib and the calibration files; and the stock packages its boot runs (image_proc,
# imu_complementary_filter, web_video_server). The board and GPIO hardware are simulated
# at run time (wire/robots/sim_libs), so the vendor code runs unchanged.
ARG BASE
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-noetic-imu-complementary-filter ros-noetic-web-video-server ros-noetic-image-proc \
      ros-noetic-nodelet ros-noetic-joy ros-noetic-dynamic-reconfigure ros-noetic-message-generation \
      libeigen3-dev python3-opencv python3-serial python3-scipy \
    && rm -rf /var/lib/apt/lists/*
ENV AINEX_REV=e8fe2a816797cf83054135160df5a82ec3596a69
RUN mkdir -p /opt/rsim_src /home/ubuntu/ros_ws/src && cd /opt/rsim_src && git init -q ainex && cd ainex \
 && git remote add origin https://github.com/Hiwonder/ainex.git \
 && git fetch -q --depth 1 origin ${AINEX_REV} && git checkout -q FETCH_HEAD \
 && mkdir -p /home/ubuntu/ros_ws/src/ainex_driver /home/ubuntu/ros_ws/src/third_party \
 && cp -r src/ainex_interfaces src/ainex_calibration /home/ubuntu/ros_ws/src/ \
 && cp -r src/ainex_driver/ros_robot_controller src/ainex_driver/ainex_kinematics \
          src/ainex_driver/ainex_sdk /home/ubuntu/ros_ws/src/ainex_driver/ \
 && cp -r src/third_party/ros-sensor_msgs_ext src/third_party/imu_calib /home/ubuntu/ros_ws/src/third_party/ \
 && source /opt/ros/noetic/setup.bash && cd /home/ubuntu/ros_ws \
 && catkin_make -DCMAKE_BUILD_TYPE=Release > /tmp/build.log 2>&1 || (tail -60 /tmp/build.log; exit 1)
RUN mkdir -p /home/ubuntu/software/ainex_controller/ActionGroups /home/ubuntu/.ros/camera_info
LABEL org.rsim.wire=ainex org.rsim.ainex=e8fe2a816797cf83054135160df5a82ec3596a69
