# robot_console

Keyboard teleoperation for a [myAGV], over rosbridge. Teleop also drives a Hiwonder
AiNex — see [`--robot`](#--robot).

The console is an independent project. Its base dependencies are `numpy`,
`opencv-python`, and `roslibpy`, so it installs and runs on a machine that has never
seen MuJoCo, MolmoSpaces, or the `simulator/` checkout. It speaks the myAGV ROS
interface and the simulator's generic arm protocol over websockets. Optional arm and
Inspect Robots dependencies do not introduce a simulator import.

## Quick start

Start a robot in one terminal:

```bash
cd ../simulator
./run.sh view --robot myagv --scene ithor:1 --ros-port 9090
```

Drive it from another:

```bash
cd robot_console
./bin/teleop.sh                        # ws://127.0.0.1:9090
./bin/teleop.sh --host 192.168.1.42    # a real myAGV on the network
./bin/teleop.sh --record runs/drive1   # ...writing feed.mp4 + commands.jsonl
./bin/teleop.sh --robot ainex          # a different robot; see below
```

The first run creates `.venv` and installs the package; after that `bin/teleop.sh` is
just a launcher. It re-installs by itself when `pyproject.toml` changes, and
`--reinstall` forces it.

Before connecting it checks that something is listening, and prints how to start each
kind of robot when nothing is. `--no-preflight` skips the check.

## Arm control

For non-ROS arms, the simulator hosts a generic msgpack-numpy observation/action
server and this project supplies the control client:

```bash
# terminal 1
cd ../simulator
./run.sh view --robot so101 --control-port 8000

# terminal 2
cd ../robot_console
uv pip install -e '.[arm]'
.venv/bin/robot-console-arm --controller wave --port 8000
```

`hold` is the safe default. A custom controller is an `observation -> action` callable
selected with `--controller package.module:function`; `example_arm_controller.py`
contains a worked example.

The SO-101 Inspect Robots integration is also entirely console-side:

```bash
uv pip install -e '.[inspect]'  # requires Python 3.10+
.venv/bin/robot-console-inspect-so101 --smoke --port 8000
.venv/bin/robot-console-inspect-so101 --port 8000  # local Ollama evaluation
```

It injects `SimulatorSO101` as the unmodified `inspect-robots-so101` hardware driver.
The simulator knows nothing about Inspect Robots, Ollama, prompts, or control policy.

### `--robot`

Two robots, two wire contracts. The keys, the loop and the recording format are the same
for both; what changes is the speed envelope, the on-screen wording and what goes on the
wire.

| `--robot` | Contract | Speeds | Turn cap | Start it with |
|---|---|---|---|---|
| `myagv` (default) | `/cmd_vel` + `/odom` | 0.05 – 0.28 m/s, step 0.05 | 1.00 rad/s | `./run.sh view --robot myagv --scene ithor:1 --ros-port 9090` |
| `ainex` | `/walking/set_param` + `/walking/command` | 0.02 – 0.20 m/s, step 0.02 | 1.00 rad/s | `./run.sh view --robot ainex --scene ithor:1 --ros-port 9090` |

Each envelope is its own hardware's, not a house default: the myAGV's is
`myagv_teleop.py`'s, and the AiNex's falls out of its gait limits — 4A/T with A ≤ 0.02 m
at a 400 ms period is 0.20 m/s and no more. Rehearsing a drive in the simulator is only
useful if the simulator refuses what the hardware would refuse.

The AiNex is the one that is not a Twist robot. It has **no `/cmd_vel`, no `/odom` and no
`/tf`** — walking is a parameter block plus a `start`/`stop` service, so
`robot_console/ainex_link.py` turns each velocity intent into gait parameters, the status
line shows `odom n/a` rather than pretending a stream dropped, and `--cmd-topic` /
`--odom-topic` do not apply.

## Controls

The OpenCV camera window owns keyboard input, which avoids a global keyboard hook.
**The window must have focus for keys to register.**

| Key | Action |
|---|---|
| `W` / `S` | Forward / back |
| `A` / `D` | Strafe left / right |
| `Q` / `E` | Rotate left / right |
| `Space` | Stop |
| `+` / `-` | Adjust speed |
| `H` or `?` | Show/hide the on-screen hints |
| `Esc` | Quit |

The same list is drawn over the live feed, so the keys are visible while driving. `H`
collapses it to a small badge.

**Hold a key to drive; let go and the robot stops.** There is no key-up event to work
with -- `cv2.waitKey` reports key-down only, and a real one would mean the global hook
this design rules out. What the OS does give is auto-repeat: holding `W` delivers `w`
over and over. So a motion is armed by a key press and expires `--hold-timeout` seconds
(0.6 by default) after the last repeat, which is what a release looks like.

That timeout has to clear the OS's *initial* repeat delay or a held key would stutter:
move, expire, then resume once repeat kicks in. macOS ships 375 ms before the first
repeat and 90 ms between them, so 0.6 s has margin while costing about 9 cm of coast at
the default speed. The vendor's own teleop makes the same trade at 0.52 s. If your
keyboard repeat is disabled or unusually slow, raise `--hold-timeout`, or pass `--latch`
to keep the old behaviour where a direction persists until `Space` or another key.

`+`/`-` step the speed by 0.05 m/s between 0.05 and 0.28; `=` and `_` work too, since
`+` and `_` need shift on most layouts.

Closing the window with its close button quits as cleanly as `Esc` does.

## Speeds

| | Value | Source |
|---|---|---|
| Default | 0.15 m/s | conservative indoor pace |
| Range | 0.05 – 0.28 m/s | 0.28 is the real myAGV's top speed |
| Turn rate | `speed x 2`, capped at 1.0 rad/s | the vendor teleop pairs 0.25 m/s with 0.5 rad/s and caps turn at 1.0 |

One knob scales the whole envelope, so a drive rehearsed in the simulator behaves the
same on hardware. `--max-speed` raises the cap and warns when it goes above the
hardware limit.

## Recording

`--record <dir>` writes two files.

`feed.mp4` is the raw camera feed, without the key hints drawn on it. The writer opens
lazily: the first 20 frames are
buffered, the true frame rate is measured from their arrival times (median interval, so
one network stall does not halve the playback speed), and only then is the file created
with the real size and rate. If no camera frames ever arrive, no `feed.mp4` is written
-- an empty file would be worse than an absent one.

`commands.jsonl` is one JSON object per line: a `meta` line, then `cmd`, `frame`, and
`odom` lines carrying `t` in seconds from the start, then a `summary` line.

```jsonc
{"type":"meta","schema":1,"host":"127.0.0.1","port":9090,"speed":0.15,...}
{"type":"cmd","t":0.05,"seq":1,"key":"w","action":"FORWARD","speed":0.15,
 "linear":{"x":0.15,"y":0.0,"z":0.0},"angular":{"x":0.0,"y":0.0,"z":0.0}}
{"type":"frame","t":0.06,"index":0,"header_seq":5880,"width":640,"height":480}
{"type":"odom","t":0.07,"x":4.68,"y":0.62,"yaw":1.03,"vx":0.15,"vy":0.0,"wz":0.0}
{"type":"summary","t":9.01,"duration":9.01,"commands":181,"frames":178,"fps":19.8}
```

`cmd` lines carry the literal `geometry_msgs/Twist` sub-dicts, so a replayer can feed a
line straight back to `/cmd_vel` without translating anything. `frame.header_seq` and
`frame.index` line the video up against the commands.

## ROS contract

| Direction | Topic | Type | Fields used |
|---|---|---|---|
| console -> robot | `/cmd_vel` | `geometry_msgs/Twist` | `linear.x`, `linear.y`, `angular.z` |
| robot -> console | `/odom` | `nav_msgs/Odometry` | pose, twist; `odom` -> `base_footprint` |
| robot -> console | `/camera/image_raw/compressed` | `sensor_msgs/CompressedImage` | base64 JPEG |
| robot -> console | `/scan` | `sensor_msgs/LaserScan` | `ranges`, angles, `range_min`/`range_max` |

`/scan` is the YDLidar X2's topic: `laser_frame`, 0.1-12.0 m, 10 Hz, mounted 65 mm ahead
of and 80 mm above `base_footprint` (`myagv_active.launch`'s static transform). **No
console command consumes it** since the mapping subsystem was removed -- but it is part
of the contract, both the real robot and the simulator publish it, and
`RobotLink.subscribe_scan` still delivers it to any caller that wants one.

Unlike `CompressedImage`, `ranges` arrives as a **plain JSON float array** -- rosbridge
base64-encodes `uint8[]` only. A no-return is reported three different ways depending on
who is publishing: `0.0` by the real driver (`invalid_range_is_inf: false`), `inf` by a
stock one, and `range_max + 1` by the simulator, which cannot express infinity in JSON.
A client must test `range_min <= r <= range_max`, which rejects all three and `NaN`
too; `tests/test_link_roundtrip.py` pins that encoding down.

Body frame, ROS convention: `+x` forward, `+y` left, `+z` counter-clockwise. The base is
holonomic -- the myAGV is Mecanum-wheeled, so `linear.y` is a real strafe, not a
no-op.

Topic names are sent absolute. The simulator's bridge normalises relative names, but
stock `rosbridge_suite` resolves them against the node namespace, where a relative name
silently misses.

`format` is matched on containing `jpeg`, because the simulator sends `"jpeg"` while a
real `image_transport` republisher sends `"rgb8; jpeg compressed bgr8"`.

### Running against a real myAGV

On the AGV:

```bash
roslaunch myagv_odometry myagv_active.launch      # odometry + the YDLidar: /odom, /scan
roslaunch rosbridge_server rosbridge_websocket.launch
# plus a camera publisher for /camera/image_raw/compressed
```

`myagv_active.launch` already starts the lidar (it includes
`ydlidar_ros_driver/launch/X2.launch`) and publishes the `base_footprint -> laser_frame`
transform, so `/scan` needs nothing extra. Then `./bin/teleop.sh --host <agv-ip>`.

**The real myAGV has no command watchdog.** `myagv_odometry_node` stores the last Twist
it received in a global and writes it to the motors at 100 Hz forever, so a robot told
to move keeps moving until it is told otherwise -- the vendor's own teleop guards
against this with a 0.52 s client-side key timeout. The console therefore treats
stopping as part of quitting rather than as best-effort: it publishes a zero Twist on
`Esc`, on window close, on an exception, and on `SIGINT`/`SIGTERM`. The simulator's
0.5 s bridge watchdog makes this invisible there; on hardware it is the only thing that
stops the robot.

## Checks

```bash
uv pip install -e '.[dev]'
.venv/bin/python -m pytest              # offline; no robot, no display

.venv/bin/python -m robot_console.smoke # live; drives the robot ~0.3 m each way
```

The offline suite covers the keymap and latch semantics, the speed model, JPEG decode
and its corruption cases, the frame mailbox under concurrency, odometry quaternion
maths, the recorder schema and video output, preflight, and CLI parsing. It also
round-trips `RobotLink` against `tests/fake_bridge.py`, a small independent rosbridge
implementation, which proves the bytes `roslibpy` emits are the bytes the server
accepts -- without needing the simulator checkout.

`smoke` is the live version: it connects, measures the `/odom` and camera rates, decodes
a frame and checks it is not a flat buffer, then drives forward, back, sideways and
around, checking the pose moved each time. It also stops publishing for 1.5 s to prove
the simulator's watchdog fires (skipped against a robot that has none). `--json` emits
the same results as one object.

## Layout

```
bin/teleop.sh              venv bootstrap + launcher
src/robot_console/
  topics.py                topic names and type strings -- the contract in one place
  teleop.py                keymap, latch state, speed model   (pure)
  camera.py                CompressedImage decode + the frame mailbox
  bridge.py                RobotLink, and pure odometry parsing
  robots.py                --robot: one RobotProfile per robot, resolved lazily
  ainex_link.py            the AiNex's gait, behind a RobotLink-shaped API
  recorder.py              feed.mp4 + commands.jsonl
  preflight.py             reachability probe + startup instructions
  app.py                   the teleop loop
  cli.py                   argument parsing
  smoke.py                 live integration check
tests/                     offline suite + fake_bridge.py
```

Everything with interesting behaviour is pure and tested; `app.py` is wiring. The loop
runs on the main thread and does everything -- read a key, publish `/cmd_vel` at 20 Hz,
decode a frame, draw it. `cv2.imshow` must own the main thread on macOS, and a separate
publisher thread would keep the robot driving while the UI was wedged. With one loop, a
stalled UI stops feeding the command stream, so a freeze degrades into a stop.

## Limitations

- Releasing a key is inferred from OS auto-repeat stopping, so the robot coasts for up
  to `--hold-timeout` after you let go. A keyboard with repeat disabled will stutter;
  `--latch` is the fallback.
- Real myAGV hardware has not been tested here; only the simulator path has been run.
- No TF, services, or parameters -- the console uses four topics. The bridge does not
  carry TF, so any sensor mount offset a client needs has to come from a constant of
  its own rather than from `/tf`.
- `mp4v` is the recording codec; `avc1` is missing from many `opencv-python` builds. If
  the writer cannot open, the drive continues and `commands.jsonl` is still written.

[myAGV]: https://shop.elephantrobotics.com/collections/myagv-smart-navigation-robot/products/myagv-pi
