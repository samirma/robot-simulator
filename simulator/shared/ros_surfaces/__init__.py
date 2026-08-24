"""Per-robot ROS surfaces, shared by every MuJoCo engine.

A *surface* is the set of topics one robot's vendor stack presents, plus the loop that
feeds them from a running MuJoCo model. It belongs to the robot rather than to any
engine: the myAGV speaks `cmd_vel`/`odom`, the AiNex speaks `/walking/*` and shares not
one topic with it, and both must look identical to `robot_console` no matter which
engine is hosting them. Keeping the implementation here is what makes that true by
construction instead of by two engines agreeing to stay in step.

The transport (`contracts/`) and the MuJoCo→wire helpers (`mujoco_bridge`) are the
layers below; a surface is the robot-specific wiring between them.
"""
