# myAGV wire image: the Noetic wire base plus the vendor's own robot_pose_ekf (a modified copy
# that publishes nav_msgs/Odometry), built from the pinned myagv_ros revision.
ARG BASE
FROM ${BASE}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends liborocos-bfl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /opt/rsim_src /opt/rsim_ws/src && cd /opt/rsim_src \
 && git init -q myagv_ros && cd myagv_ros \
 && git remote add origin https://github.com/elephantrobotics/myagv_ros.git \
 && git fetch -q --depth 1 origin c71f3cc574e5ed1973a925238eabe88662cfa701 \
 && git checkout -q FETCH_HEAD \
 && cp -r robot_pose_ekf /opt/rsim_ws/src/ \
 && source /opt/ros/noetic/setup.bash && cd /opt/rsim_ws \
 && catkin_make -DCMAKE_BUILD_TYPE=Release > /tmp/build.log 2>&1 || (tail -40 /tmp/build.log; exit 1)
LABEL org.rsim.wire=myagv org.rsim.myagv_ros=c71f3cc574e5ed1973a925238eabe88662cfa701
