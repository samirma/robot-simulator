# CoppeliaSim engine

CoppeliaSim (formerly V-REP) as a **simulator engine** for this workspace: it hosts a
scene with one of the shared robots built into it, and presents that robot on the *exact
same wire contract* as the MolmoSpaces and RoboCasa engines and the real hardware.
`robot_console` connects to a CoppeliaSim-hosted robot with no changes — that is the whole
point of the `simulator/shared/` split.

## Quick start

```bash
./run.sh setup                                           # venv + ZMQ client + CoppeliaSim EDU (~260 MB)
./run.sh view --robot myagv --scene room:6 --ros-port 9090
./run.sh view --robot so101 --scene empty --control 127.0.0.1:8000
./run.sh shell
```

`--scene`:
- `room[:<size_m>]` — a square walled room of the given inner size (default 6 m). The
  myAGV's laser sees these walls. Always available.
- `empty` — no walls (an SO-101 bench arm needs none).

`--robot`: `so101` (arm) or `myagv` (holonomic mobile base). Both come from
`simulator/shared/robots/` — the same specs the other engines use (the SO-101 from its
shared URDF, the myAGV as an idealized holonomic base).

The wire surface is mutually exclusive, mirroring the other engines:
- `--ros-port N` — present the myAGV vendor ROS topics (`/cmd_vel`, `/odom`, `/scan`,
  `/camera/image_raw/compressed`) over rosbridge. Mobile base only.
- `--control HOST:PORT` — present the generic msgpack-numpy arm control server
  (`molmospaces-control-v1`). Used by `robot_console`'s arm client / SO-101 driver.
- `--headless` — best effort only; see the macOS note below.

## How it works

CoppeliaSim is driven entirely over its **ZMQ remote API** (no scene `.ttt` authoring and
no in-sim child scripts). `tools/spawn_robot.py`:

1. Launches CoppeliaSim into the macOS GUI session with `open` (see the note below),
   waits for the ZMQ server, and connects.
2. Builds the scene and robot programmatically: room walls as primitive cuboids; the myAGV
   as a kinematic box base carrying an RGB vision sensor; the SO-101 imported from the
   shared URDF (`simURDF.importFile`) with its joints switched to kinematic mode.
3. Runs a `sim.step()` loop at the control rate, feeding the **shared** wire transports in
   `simulator/shared/contracts/` — `rosbridge_server.py` for the myAGV, `control_server.py`
   for the SO-101. These are the same servers every engine uses, so the console cannot tell
   the engines apart.

Because the wire transport and the message builders are shared code, a CoppeliaSim myAGV
publishes `/odom`+`/scan`+camera and obeys `/cmd_vel` identically to a MolmoSpaces or
RoboCasa myAGV or the real 2023 Pi myAGV; a CoppeliaSim SO-101 speaks the identical control
protocol (`molmospaces-control-v1`, move groups `arm`:5 / `gripper`:1).

Unlike the MuJoCo engines, this adapter does **not** import `simulator/shared/mujoco_bridge.py`
(that helper imports `mujoco`, which is deliberately absent from this engine's venv). The
CoppeliaSim-specific sourcing lives in `spawn_robot.py` instead:

- **myAGV base** — driven *kinematically*: `/cmd_vel` (body frame; `+x` fwd, `+y` left, the
  real Mecanum base's genuine strafe) is integrated into a world pose and applied with
  `setObjectPose`. This matches the idealized holonomic base the MuJoCo engines use. Odom
  twist reports the commanded velocity (post-watchdog), the same choice `serve_ros` makes.
- **laser `/scan`** — computed analytically against the room wall segments rather than from
  a ring of proximity sensors. Deterministic, cheap over the remote API, and reproduces the
  YDLidar X2 contract exactly (360 beams CCW from `-pi`, 0.1–12 m, 10 Hz, misses as
  `range_max + 1`, 65 mm forward laser mount).
- **camera** — a genuine CoppeliaSim RGB vision sensor. Its image is bottom-up (OpenGL
  origin) and is flipped before JPEG encoding.
- **SO-101 joints** — imported from the shared URDF and set to kinematic mode, so a
  commanded joint position is tracked exactly (a position-controlled arm, which is what the
  control contract assumes).

## Layout

```
coppeliasim/
  run.sh env.sh
  tools/spawn_robot.py     launch CoppeliaSim + build robot + shared-bridge serve loop
  app/                     CoppeliaSim.app (downloaded by setup; gitignored)
  .venv/ data/             (gitignored)
```

There is no per-robot adapter package: the shared URDF/spec drives the CoppeliaSim build
directly, so the robot definitions live only in `simulator/shared/robots/`.

## Notes

- **macOS launch:** CoppeliaSim must reach the WindowServer, so it is started with `open`
  (which routes it into the Aqua GUI session). A plain `exec` of the binary from a non-GUI
  shell — and, for the same reason, `-h` "headless" mode on a host with no GUI session —
  exits immediately after startup. `--headless` is therefore best effort; on a normal
  desktop session `view` opens the CoppeliaSim window.
- **CoppeliaSim's own Python:** the ZMQ *client* is pure Python and lives in this engine's
  venv. Separately, CoppeliaSim's *sandbox* Python (the interpreter it runs internally)
  needs `pyzmq` and `cbor2`; `run.sh setup` installs both into the venv and points
  CoppeliaSim's `defaultPython` at it (`~/.CoppeliaSim/usrset.txt`). Without this the
  sandbox logs a "could not handle the wrapper script" error.
- **Gatekeeper:** the EDU `.app` is ad-hoc signed; `run.sh setup` clears the quarantine
  attribute (`xattr -dr com.apple.quarantine`), otherwise macOS blocks the ZMQ/URDF plugin
  dylibs with confusing load failures.
- **No MuJoCo:** this engine's venv has no `mujoco`; the adapter reuses only the MuJoCo-free
  `contracts.*` transports from `simulator/shared/`.
