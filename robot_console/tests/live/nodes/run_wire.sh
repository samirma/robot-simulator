#!/bin/bash
# Container entry: stock rosbridge_websocket (+ rosapi) on :$PORT (default 9090) plus the test wire node.
#   run_wire.sh <spec.json>
set -e
source /opt/ros/$ROS_DISTRO/setup.bash
if [ "$ROS_VERSION" = "1" ]; then
  roscore >/tmp/roscore.log 2>&1 &
  until rostopic list >/dev/null 2>&1; do sleep 0.2; done
  roslaunch --wait rosbridge_server rosbridge_websocket.launch port:=${PORT:-9090} >/tmp/rosbridge.log 2>&1 &
  exec python3 -u /nodes/ros1_wire.py "$1"
else
  ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=${PORT:-9090} >/tmp/rosbridge.log 2>&1 &
  exec python3 -u /nodes/ros2_wire.py "$1"
fi
