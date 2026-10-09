# reBot Arm B601-RS wire image: the Jazzy wire base plus Seeed's ROS 2 workspace as the B601-RS
# ROS 2 wiki builds it, at /opt/rsim_ws: the pinned Seeed-Projects/reBotArmController_ROS2
# revision with rebotarm_msgs, rebotarmcontroller and rebotarm_bringup built by colcon (the
# packages the authoritative boot runs), and the pinned Seeed-Projects/reBotArm_control_py SDK
# in third_party/reBotArm_control_py, where the controller looks for it; and the ROS packages
# the wiki installs for them (Pinocchio, tf_transformations, control_msgs, trajectory_msgs;
# rviz2 only serves the optional use_rviz:=true). The CAN driver motorbridge is not installed:
# the motors are simulated at run time (wire/robots/sim_libs/motorbridge.py), so the vendor
# code runs unchanged. Every package comes from the base image's dated snapshots (ROS:
# snapshots.ros.org jazzy 2026-09-11; Ubuntu: snapshot.ubuntu.com noble 20261008T213000Z);
# the interface file records no release for any of them.
ARG BASE
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-pinocchio ros-jazzy-tf-transformations python3-transforms3d \
      ros-jazzy-control-msgs ros-jazzy-trajectory-msgs ros-jazzy-geometry-msgs \
      ros-jazzy-rosidl-default-generators ros-jazzy-rosidl-default-runtime \
    && rm -rf /var/lib/apt/lists/*
ENV REBOT_ROS2_REV=7506dc0fc5c2a7787dcb974d89d13f9f49d16d61 \
    REBOT_SDK_REV=6415d43130d1e143c70dc106096a857ac5556f81
RUN git init -q /opt/rsim_ws && cd /opt/rsim_ws \
 && git remote add origin https://github.com/Seeed-Projects/reBotArmController_ROS2.git \
 && git fetch -q --depth 1 origin ${REBOT_ROS2_REV} && git checkout -q FETCH_HEAD \
 && git init -q third_party/reBotArm_control_py && cd third_party/reBotArm_control_py \
 && git remote add origin https://github.com/Seeed-Projects/reBotArm_control_py.git \
 && git fetch -q --depth 1 origin ${REBOT_SDK_REV} && git checkout -q FETCH_HEAD \
 && cd /opt/rsim_ws && source /opt/ros/jazzy/setup.bash \
 && colcon build --packages-select rebotarm_msgs rebotarmcontroller rebotarm_bringup \
      > /tmp/build.log 2>&1 || (tail -60 /tmp/build.log; exit 1)
RUN source /opt/ros/jazzy/setup.bash && source /opt/rsim_ws/install/setup.bash \
 && python3 -c "import pinocchio, tf_transformations, rebotarmcontroller.hardware_config, rebotarm_msgs.msg" \
 && test -f /opt/rsim_ws/install/rebotarm_bringup/share/rebotarm_bringup/description/RS/urdf/ReBot_Arm_RS.urdf
LABEL org.rsim.wire=rebot_b601_rs org.rsim.rebotarmcontroller_ros2=7506dc0fc5c2a7787dcb974d89d13f9f49d16d61 \
      org.rsim.rebotarm_control_py=6415d43130d1e143c70dc106096a857ac5556f81
