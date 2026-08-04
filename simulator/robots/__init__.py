"""Out-of-tree robot definitions for the MolmoSpaces simulator.

Each subpackage supplies an MJCF plus the move groups, RobotView, Robot and
BaseRobotConfig subclasses MolmoSpaces needs. Because `BaseRobotConfig.robot_dir`
accepts an external directory, none of this requires forking molmospaces.

See molmospaces/docs/tutorials/add_robot.md.
"""
