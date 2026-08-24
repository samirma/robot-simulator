"""Occupancy-grid SLAM for the myAGV, built on `/scan`.

The same choice Elephant Robotics makes: `myagv_navigation/launch/myagv_slam_laser.launch`
runs gmapping on the YDLidar X2, and `navigation_active.launch` navigates the `map_server`
pgm/yaml pair it produces. This package reads the same topic and writes the same map
format, so a map is portable in both directions between the simulator and the robot.

What it is not: a full graph SLAM. There is no loop closure and no global optimiser --
those need a solver, and a house-sized map that is locally consistent and arrives inside
the control loop's budget is worth more here than a globally optimal one that does not.

Layout, in dependency order:

    scan        LaserScan messages -> base-frame points
    grid        the log-odds occupancy map, and how a scan folds into it
    mapio       map.pgm + map.yaml (+ npz sidecar) read and write
    matcher     correlative scan matching against a likelihood field
    pose        PoseTracker: odom propagation + keyframed matching
    planner     obstacle inflation and A*
    frontier    where the map stops, and which gap to fill next
    controller  path -> /cmd_vel for a holonomic base
    mapview     rendering, and clicks back into metres
    app         the shared loop the three commands are modes of
"""

from __future__ import annotations

__all__ = [
    "controller",
    "frontier",
    "grid",
    "mapio",
    "mapview",
    "matcher",
    "planner",
    "pose",
    "scan",
]
