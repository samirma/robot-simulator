"""The myAGV + myCobot 280's ROS contract -- the console's copy of what it uses.

The authority is `robots_specs/myagv_mycobot280/ros.yml`, which `extends` the myAGV's:
the composite's ROS graph is the myAGV's boot launch plus Elephant's composite navigation
launch (`map_server`, `amcl`, `move_base`). The myCobot 280 on its deck is driven over a
pymycobot socket, not ROS, so nothing here concerns it. The simulator transcribes the
additions in `simulator/shared/ros_surfaces/myagv_mycobot280.py`; the workspace parity
tests hold this copy equal to it.

The base is a myAGV: every name and type in `topics.py` applies unchanged, as do its
speeds. What differs is the stop command -- `move_base` keeps sending velocities while a
goal is active, so stopping cancels navigation first (`TOPIC_CANCEL`, an empty GoalID),
then sends the zero Twist -- and what tells it apart on a wire: `move_base`'s action goal
topic, `TOPIC_GOAL`.
"""

from __future__ import annotations

#: move_base's actionlib goal: what identifies the composite among `/cmd_vel` bases.
TOPIC_GOAL = "/move_base/goal"
TYPE_GOAL = "move_base_msgs/MoveBaseActionGoal"
#: Where the stop command cancels navigation, and the empty GoalID it sends (an empty id
#: and a zero stamp cancel every goal).
TOPIC_CANCEL = "/move_base/cancel"
TYPE_GOAL_ID = "actionlib_msgs/GoalID"
CANCEL_ALL = {"stamp": {"secs": 0, "nsecs": 0}, "id": ""}
