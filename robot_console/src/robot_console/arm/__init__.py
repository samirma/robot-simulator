"""The SO-101 arm task: an Inspect Robots task, policies, embodiment and scorers.

This subpackage is the console's arm half, and it is deliberately walled off from the
rest of `robot_console`. Everything here talks to a simulator (or a real arm) over
**rosbridge**, on the topic set in `simulator/shared/ros_surfaces/so101.py`; nothing here
imports from `simulator/`, and the only runtime dependencies are numpy plus the
`inspect-robots` framework. Torch lives behind a function-local import in `molmoact`, so
the offline test suite collects and runs without it.

Nothing is imported eagerly here. The modules are reached through the
`inspect_robots.{tasks,policies,embodiments,scorers}` entry points in `pyproject.toml`,
which is what lets `inspect-robot run --task apple_on_plate --policy molmoact2
--embodiment so101_ros` find them; importing them from this file would drag the whole
graph in for `import robot_console`.
"""
