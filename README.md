# Robot Simulator

A robot simulation workspace with two independent projects that talk over network protocols (rosbridge), never Python imports:

- **`simulator/`** — simulates robots (myAGV mobile base, SO-101 arm, AiNex humanoid) with a choice of two interchangeable MuJoCo engines: MolmoSpaces (iTHOR houses) and RoboCasa (kitchens). Both engines expose each robot's real vendor ROS interface over one rosbridge websocket.
- **`robot_console/`** — drives, maps and navigates the simulated (or physical) robots: teleop, SLAM exploration, and the SO-101 arm task. It connects identically regardless of which engine hosts the robot.

## Quick start

Serve a simulated kitchen with robots on `ws://127.0.0.1:9090`:

```bash
cd simulator
./kitchen.sh serve                       # headless, SO-101 arm task
./kitchen.sh serve --engine robocasa     # the other engine
./kitchen.sh serve --robots so101,myagv  # a fleet, one port
./kitchen.sh serve --mujoco              # ...with a MuJoCo window
./kitchen.sh view                        # live camera page
```

Drive and task the robots from the console:

```bash
cd robot_console
./bin/teleop.sh          # keyboard teleop
./bin/slam.sh explore    # autonomous mapping
./run_task.sh            # run the arm task with the VLA policy
```

See `CLAUDE.md` for the full architecture, commands and invariants, and each engine's own `README.md` for per-engine details.
