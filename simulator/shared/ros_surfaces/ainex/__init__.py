"""The AiNex's ROS contract: the manufacturer's own topics, in one shared place.

A package rather than a module, which is not the shape the other two robots needed and is
inherent to this one: the myAGV's contract is a Twist in and odometry out, while a Hiwonder
AiNex is a 24-servo walking state machine with a vendor action library. The pieces are
`surface.py` (the loop), `topics.py` (the vendor names), `servos.py` (the bus servo table),
`gait.py` (the walk) and `actions.py` with its `action_groups/` data. All of them are
engine-neutral -- none imports `molmo_spaces` -- which is what let them move here.

It lives in `shared/` for the same reason the myAGV's does: the topic set a robot presents
belongs to the robot, not to whichever engine happens to be simulating it, and two copies
of this loop would be two chances for the engines to drift. See `simulator/CLAUDE.md`.

`servos.py` sits here rather than beside the robot's spec because `shared/robots/` is
deliberately not importable -- each engine already has a `robots` package on its path that
would shadow it (`shared/robots_spec.py` says so). It is the vendor servo table, shared by
the surface that serves it and by whatever generates the model from it.
"""

from __future__ import annotations

from ros_surfaces.ainex.surface import attach_ros, serve_ros

__all__ = ["attach_ros", "serve_ros"]
