# Simulator checks

Run with an engine venv's python (MuJoCo, NumPy, PyYAML; `pytest` and `pytest-timeout` are
installed into it by the commands below if missing):

```sh
VIRTUAL_ENV=simulator/molmospaces/.venv uv pip install pytest pytest-timeout
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/unit simulator/tests/integration
simulator/molmospaces/.venv/bin/python -m pytest simulator/tests/e2e --timeout 3600   # Docker; ~1 h
```

Every scene the checks use is one the reference project's `kitchen.sh` loads (MolmoSpaces
`ithor:1`, RoboCasa `robocasa:1-1`), except the cross-engine comparison, which uses the
shared `test:1` scene, and the placement unit checks, which build minimal worlds (a pen, a
bare floor, a covered worktop) to reach each refusal of the placement function.

| Suite | Checks |
| --- | --- |
| `unit/test_registry.py` | the registry is `robots_specs/high_level_spec.md`: ids, names, kinds, files, dialects, composite components, the mounting transform read from §6, the unknown-id message, required files |
| `unit/test_scenes.py` | `--scene` parsing on both engines: defaults, other engine's source named, id ranges |
| `unit/test_protocol.py` | control-port framing, request/reply/event matching, the CBOR decoder |
| `unit/test_worktop_survey.py` | the reference survey: the worktop (largest table-height top; a surface holding the task's categories first), deterministic spots inside the edges, the first spot facing the most worktop, RoboCasa's back-edge spot facing the room then the tabletop search then the front edge |
| `unit/test_worktop_objects.py` | the six objects at the reference's poses in the robot frame, the reference's clearing (radius, band, robots kept, a child goes with its parent), apply/undo leaving the spec as it was, cleared state restored, missing meshes refused |
| `unit/test_placement.py` | the worktop is the survey's surface, no worktop outside table height, an arm at the survey's first spot with its staging, repeatable spot and heading, occupied worktop, no clear spot (a robot without staging), clearing the loose objects in an arm's working area, refusal when every spot intersects the scene or the objects would not stand on the worktop, a worktop that supports a mobile robot but not its travel, floor travel and open-floor facing, camera clearance (support surface not counted) and its refusal, second robot kept clear of the first, no robot left after any refusal |
| `unit/test_world.py` | adding/removing robots keeps time, positions, velocities and controls of the survivors and of a moving scene object; a worktop robot's staging enters and leaves with it, the cleared objects come back with their state, the survivors keep theirs bit for bit; a failed add rolls back robot and staging; new robots start at home; robots collide with each other but keep their own self-collision; the physics loop keeps real time; the RTF < 0.90 warning |
| `unit/test_cli.py` | `spawn.sh`/`kitchen.sh`/`run.sh` help, unknown flags and ids, taken ports, no simulation, `--arm-port` refusal, scene refusals |
| `integration/test_simport.py` | the simulation on `ithor:1` over its control port: hello/scene, a worktop robot's staging and its removal (cleared objects back where they were), PNG/RGB/depth renders, admission (duplicate id, busy startup, reserved port, occupied worktop), release when a spawn's connection ends or its process is SIGKILLed, readings and actuator commands, state kept across another robot's spawn and removal, robot-camera renders |
| `integration/test_default_scenes.py` | each engine's default scene has the reference loader's model counts, no robot, the survey's worktop; so101 and mycobot280 at their golden spots with the six objects at their poses and the golden cleared set; the counts back after removal |
| `integration/test_reference_parity.py` | with `RSIM_REF_DIR` (a checkout of the reference at `34547ae`; skipped without it): so101 on each engine's default scene against the reference's own code (`reference/ref_driver.py`): spot xy and surface z within 1 µm, yaw within 1e-9 rad, base z within 3 mm, the six objects within 1 µm, the same cleared objects |
| `e2e/test_robots.py` | every robot on both engines' default scenes: embodiment (components, cameras, lidars, placement, at rest; an arm's six worktop objects staged, the motions run with them), the interface contract (nodes, topics and types, services and types, actions, parameters and values; only ROS infrastructure beside it), message fields and frames, periodic rates within +-10% with no gap over three periods (measured on the wire's own graph), motion smoke runs through the vendor interface (drive/walk + stop, arm, gripper, head, action group) judged by recorded feedback or control-port readings with the recorded tolerances, camera images against the simulation's render, lidar geometry |
| `e2e/test_lifecycle.py` | readiness line, exit 0 on SIGINT/SIGTERM, duplicate and busy refusals, occupied worktop, isolation of the remaining robot, the worktop objects coming and going with the arm (no `task_*` left, cleared objects back), SIGKILL release, container exit, rosbridge failure inside a running container, a vendor node failing, one of an assembly's two wires failing, a partially failed startup, the simulation ending (non-zero), no Docker, missing files |
| `e2e/test_engine_consistency.py` | each robot on both engines' `test:1`: identical interface; same drive/walk/arm behaviour within the tolerances |

`wirecheck.py` holds the interface comparison; `INTERFACE_FILE_GAPS` lists the differences
that come from unchanged stock code the boots run, not from the simulator (reported by the
checks and raised with robots_specs). Rates are measured by `rate_probe_ros1.py` /
`rate_probe_ros2.py` inside the wire container (a native subscriber on the robot's graph):
rosbridge, a Python server, cannot carry every camera stream and point cloud of a robot at
once, which would measure rosbridge, not the robot. Each probe subscribes with the
publishers' reliability and measures after a 5 s warm-up (`RSIM_PROBE_WARMUP`): the probes
joining a ROS 2 graph briefly stall its writers while discovery settles.
