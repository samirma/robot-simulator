# Robot Console — High-Level Specification

This document is the complete high-level specification for the standalone robot console.
The console neither imports nor requires robot firmware, a robot-side software checkout or a
sibling source tree. No other document or code in the console may contradict this document.

This document defines only the console's requirements, subject to the shared goal and
project boundary in [`../high_level_spec.md`](../high_level_spec.md). The designated robot
reference is [`../robots_specs/high_level_spec.md`](../robots_specs/high_level_spec.md)
together with its robot files `../robots_specs/<id>.md`, referred to below as the
**robot specification**. It
identifies robots and pinned interface sources. The console independently derives and
packages the interface facts it needs from those sources. The simulator's specification
and implementation are not authoritative for the console.

## 1. Goal

Connect to compatible physical robots over ROS 1 or ROS 2 to drive them, view their cameras
and operate the bounded controls described below. The same console works against a
simulated wire that matches a supported real-hardware interface. Mapping, navigation,
grading and recording are outside the console's scope.

Every entry point takes `--url ws://<host>:<port>` to choose a rosbridge websocket connected
to either a ROS 1 or ROS 2 graph, defaulting to `ws://127.0.0.1:9090`. The ROS dialect is
determined from the selected profile and verified against discovered names and types before
commands are enabled. ROS 1 and ROS 2 type spellings and action/service conventions are
handled by the transport layer; they do not change user-facing behaviour. The **launchers**
are the shell entry points (`teleop.sh`, `view.sh`).

| Entry point | Responsibility |
| --- | --- |
| `teleop.sh` | Keyboard teleoperation of a supported mobile robot, with all its discovered supported cameras shown live. |
| `view.sh` | Browser page showing the connected robot's cameras and read-only sensors, with per-robot controls. |
| `python -m robot_console.fleet` | Non-interactive wire check (§2.4): validate the wire's robots against the profiles and their cameras. |

### 1.1 Supported profiles and interface selection

The console owns the following profiles; these are the only accepted robot ids:

| Robot id | ROS dialect | Teleoperation |
| --- | --- | --- |
| `myagv` | ROS 1 | Base motion |
| `ainex` | ROS 1 | Walking and head control |
| `rosmaster_x3_plus` | ROS 1 | Base motion |
| `rebot_b601_rs` | ROS 2 | Not supported (arm) |

Each packaged profile is the console's normative typed interface and control catalog.
It records its pinned authoritative sources, required interface evidence for identification,
command and feedback names/types, units, limits, documented stop/cancel operations,
for each action control what its server does with a goal that arrives while another of its
goals runs (preempts the running goal, rejects the new one, or runs it beside the running one),
camera types (a camera of an optional add-on, such as a wrist camera, is itself optional),
the read-only `sensors` the view page may show (manufacturer topics, each an output topic of
the profile with its type, the message fields to show and a stale threshold, or none for a
latched topic), the per-axis teleoperation speed a held key commands (for mobile profiles),
and the exact bounded controls allowed on the page. Every catalog entry
must trace to those sources; undocumented controls or invented feedback are prohibited.
Optional endpoints gate only the capability that needs them. An exposed control requires all of its
documented prerequisites; incompatible or missing prerequisites leave it unavailable
with a visible reason. Profiles must ship with the console and be inspectable without
robot firmware or the workspace's robot asset folders.

`--robot` selects one of these profiles. When it is omitted, the console selects a profile
automatically only if discovery uniquely identifies a supported profile and target;
otherwise it refuses commands and asks for explicit selection. An explicit profile must still pass typed validation.
`teleop.sh` refuses an unknown id, an arm id, or an automatically identified arm target,
listing the teleoperable ids (`myagv`, `ainex`, `rosmaster_x3_plus`).

Use each hardware interface's documented ROS names. Namespace selection is supported
only where the pinned real-hardware sources explicitly document that interface as
namespace-configurable; such profiles record the supported name composition and expose
`--namespace <name>` in both launchers and equivalent selection on the page. For those
profiles, an omitted namespace selects the documented hardware default if that target is
present; otherwise a single discovered target is selected, and several require explicit
selection. Other profiles offer no namespace override and refuse
one rather than rebasing hardware names. Validation, subscriptions and commands always
use the same selected target, preserving documented absolute/global names.

## 2. Components

### 2.1 Teleoperation — `teleop.sh`

```sh
teleop.sh [--robot <id>] [--namespace <name>] [--url ws://…]
```

* **Keys.** The **motion keys** are:
  * `W`/`S`: forward and back;
  * `A`/`D`: strafe left and right (the wheeled bases are omnidirectional; the AiNex
    strafes through its gait);
  * `Q`/`E`: rotate left (counter-clockwise) and right (clockwise).
* **Head keys** (`ainex` only): the arrow keys move the head within its documented
  limits — Left/Right pan, Up/Down tilt — in the key's direction as seen on the robot's
  model: Left turns the head to the robot's left, which is a negative `head_pan` about the
  model's −Z axis. They are not motion keys and never request the walking stop.
* **Other keys.** `Space` stops (below). `Esc` quits as an ordinary exit. Commands are
  enabled automatically once the start-up stop has been delivered; no key is needed to
  start controlling the robot. `Enter` explicitly re-enables commands after a
  failed stop, including a failed start-up stop.
* **Motion lifecycle.** Motion is hold-to-move: release of a motion key removes its
  contribution, and release of the last motion key requests the profile's documented
  stop. Simultaneous keys combine only within documented limits; opposite directions
  cancel. `Space` clears all motion intent and requests stop; movement requires a fresh
  press afterward. Input loss or focus loss also clears intent and requests stop.
  The input facility must detect held/released keys; key repeat alone is not evidence
  that a key remains held. Head input stops changing its target on release.
* **Surface.** `teleop.sh` opens a local window that owns keyboard focus and shows the
  camera streams (§2.3). Focus loss is that window losing keyboard focus, input loss is the
  key-event source ending or failing, and closing the window is an ordinary exit.
* **Speed.** A held motion key commands its profile's recorded teleoperation speed for
  that axis, never above documented limits.
* On an ordinary exit or an interruption the process can handle, clear motion intent and
  attempt the documented stop. Show connection/stop failures. Do not claim stop delivery
  on a broken connection or uncatchable process termination; state any independently
  documented robot watchdog behaviour, or explicitly state that stopping in those cases is
  not guaranteed.
* **Connection loss** clears motion intent, shows the failure and the limitation above,
  and ends `teleop.sh` with a non-zero status; reconnecting means launching it again. On
  every start, once the target is validated and before commands are enabled, teleop
  attempts the documented stop once and shows the outcome, so a relaunch stops motion a
  lost session left running.
* A stop fails explicitly when the transport rejects or cannot send it, or when the
  documented stop service or action returns an error or times out; a publish with no
  acknowledgement is not a failure. An explicitly failed stop disables commands and clears
  held input. Re-enabling requires an explicit user action and fresh key input; old motion
  intent is never resumed.
  Re-enablement does not check or confirm that prior motion stopped. Display that
  limitation together with the failure.

### 2.2 Camera and control page — `view.sh`

```sh
view.sh [--url ws://…] [--robot <id>] [--namespace <name>]
```

* One static page speaking rosbridge to `--url`. It shows the selected target's cameras
  (§2.3), only the bounded controls its profile's catalog lists (§1.1), such as a head
  position, an arm trajectory, a gripper position or a listed bounded action, the profile's
  read-only sensors (§2.5) and, for a teleoperable profile, the base motion below. It
  subscribes to nothing but the target's cameras and its profile's sensors, and it offers no
  generic topic publication or subscription.
* Robot selection, and namespace selection where supported, follows §1.1. Without a
  validated target the page shows the reason and a robot selector, and offers no controls.
* Controls are live as soon as the target passes typed validation; there is no arm or
  enable step. Nothing is sent before validation or on load, with two exceptions once the
  target is validated: the page may poll the profile's documented read-only joint-position
  query about once per second, so that the model follows a robot that publishes no joint
  states (this poll never commands motion and stops on connection loss or target change);
  and, for a teleoperable profile, it attempts the documented stop once before base motion
  is enabled, as `teleop.sh` does at every start, and shows the outcome.
  A value the user changes is sent to the validated target immediately, except that an
  action control whose server does not preempt holds it while the control's goal runs:
  * a topic-publish control publishes while its slider, or the AiNex head pad, is dragged,
    at about 10 messages per second;
  * an action control whose server preempts its running goal sends goals while its slider
    is dragged, at about 5 per second, each preempting the previous one;
  * an action control whose server rejects a goal that arrives while another of its goals
    runs, or runs that goal beside the running one, has at most one goal outstanding (below).
    While its goal is outstanding, the page holds a changed value, keeping only the latest,
    and sends nothing. When the goal stops being outstanding, normally because its result
    arrives (whatever its status: succeeded, canceled, aborted or rejected), the held value
    is sent as the next goal if it differs from the last goal sent, and is dropped
    otherwise, so a drag delivers its final value after the running goal's result. A value
    changed while no goal is outstanding is sent as the next goal, also when it equals the
    last goal's, and replaces a held value still waiting to be sent. As for a server that
    preempts, the control sends goals no more often than about 5 per second: a goal due
    sooner after the previous one waits for that pacing. The page never cancels the running
    goal to make room for a newer one. A goal that rosbridge reports the server rejected is
    shown as rejected, with that reason, and is not resent.
* The page supports performing actions, as `teleop.sh` does. For a teleoperable profile
  (`myagv`, `ainex`, `rosmaster_x3_plus`) it offers `teleop.sh`'s base motion and AiNex head
  keys (§2.1) with the same profile speeds, limits and stop: the keys act while the page,
  not an editable field, has keyboard focus, and on-page hold buttons do the same for
  pointer and touch input. Motion is hold-to-move with §2.1's lifecycle: release of the last
  motion input, `Space` or a Stop button, focus loss, and page hide or unload clear motion
  intent and request the documented stop; a target change does so before leaving the old
  target. An explicitly failed stop disables base motion until an explicit re-enable action
  and fresh input, with §2.1's stated limitations. Connection loss clears motion intent and
  states that no stop could be delivered. An arm profile gets no base motion.
* The page shows a 3D model of the selected target's embodiment in its current pose,
  updated from the joint positions the robot reports: when an arm joint moves, the same
  joint in the model turns by the same angle. Each movable joint that a catalog control
  commands can be clicked in the model to change that control's value. The model is built
  from the robot's URDF.
* Every joint and topic that a control commands can be edited on the page and shows the
  robot's current value, where the profile documents feedback for it.
* Every control commands a single finite target. Values are clamped to the documented
  limits, and a value outside them is shown as invalid and never sent. Where the profile
  documents measured joint positions, the page shows them and can copy them into the
  targets. Each control shows its sending, done and failed state, and an action control
  also its running and rejected states.
* Apart from base motion, the page never stops or cancels anything: no bounded control has
  a stop button or key, and the page sends no stop or cancel for one on release, focus
  loss, page hide or unload, target change or connection loss, or to make room for a newer
  goal. A goal keeps running if the tab is closed. The page states that changes are sent
  immediately and that only base motion is stopped automatically, and how an action
  control whose server does not preempt holds changes while its goal runs.
* ROS 2 action goals use one rosbridge connection each, because stock rosbridge
  serializes a client's operations while its goal runs. A goal is acknowledged by its first
  feedback or its result, and is outstanding from when it is sent until its result arrives,
  its connection is lost, or it has gone 1 s without acknowledgement. The next goal of a
  control waits at most 1 s for the previous goal's acknowledgement and, where the server
  does not preempt, for the result of an acknowledged goal; a rosbridge that never answers
  a rejected goal therefore holds the next one back for 1 s only. A goal's connection
  closes after its result, a superseded goal's once the next goal is acknowledged, and at
  most three are open per control.
* Connection loss disables the controls. The page then shows the disconnected state and
  connects again only when reloaded. A target change requires new validation; apart from
  the base-motion stop above, it sends nothing to the old target and leaves goals already
  sent running.

### 2.3 Live cameras

Both entry points display all supported camera streams discovered for the selected robot
target, identified by their resolved topic names. Until a target is selected and
validated, they show every discovered supported camera stream, clearly marked as not tied
to a target. Accept the documented ROS 1 or ROS 2
raw image types and encodings supported by the profile, without substituting a simulated
image source. Each stream shows live images and a visible unavailable, unsupported,
stale or failed state when applicable. Profiles define a finite stale-image threshold
from documented rates or an explicitly identified console policy. Never present a frozen
frame as live. Having no cameras is a visible state, not a connection failure for
independent controls. A camera the profile marks optional that is not on the wire is shown
as unavailable and does not fail validation. A profile lists only the streams the page
shows. A depth camera's depth stream (a raw single-channel 16-bit image, `16UC1`) is a
camera of the profile of kind `depth`, beside the camera's colour stream. The profile
gives its metres per value and its range (near, far); the stream is drawn as a
false-colour map scaled to that range (nearer, warm, to farther, cool; values beyond the
range take the end colours), a value of 0 (no depth) is black, and the page shows a
legend with the range and the black "no depth" swatch. A depth camera documents `16UC1`
only; another encoding is an unsupported frame. Both entry points and the wire check
decode a depth frame with that scale, and a 16-bit stream with no scale (an untied one)
is stretched to its own maximum. Exiting closes subscriptions and the connection.

### 2.4 Wire check — `python -m robot_console.fleet`

```sh
python -m robot_console.fleet [--url ws://…] [--expect <id>]
```

* Discover the robots on the wire through typed signatures and check each against its
  profile: every required endpoint with its type, and nothing else under its names but the
  profile's optional rows and ROS infrastructure (workspace spec §1.3). `--expect` narrows
  this to one robot id, which must present its whole typed interface. Every ambiguous
  case fails, naming the candidates found.
* Report each profile camera as live, stale or missing (§2.3). The check publishes nothing
  and ends with a non-zero status when validation fails, an expected camera is not live, no
  supported robot is discovered, or the wire is unreachable. A missing or not live optional
  camera is reported as optional and does not fail the check.

### 2.5 Sensor panel

The view page shows the selected target's profile `sensors`: manufacturer data topics
(motor states, joint states, arm status) that the profile records as output topics, each
with its type, the message fields shown (a path into the message, a unit, and for a status
word the documented meaning of each bit) and a stale threshold, or none for a latched
non-periodic topic.

* The panel is read-only: the page subscribes to the profile's sensor topics once the target
  is validated and never publishes, calls or cancels anything for them. A profile without
  sensors shows no panel.
* Each sensor shows its resolved topic, its current field values, the age of its last
  message and a state: no message yet, live, stale (the last message is older than the
  threshold; the values are marked frozen), last value (a latched topic that has delivered,
  never stale), missing from the wire, or disconnected. Values are never presented as
  current once the state is not live or last value. A status word shows each documented bit
  as set or clear. Undocumented fields are not shown.
* Sensors share the page's single connection and are unsubscribed on a target change.

## 3. Constraints

The console has these constraints:

* It installs its own venv on first run and whenever `pyproject.toml` changes.
* The shell entry points share profile selection, typed validation and ROS transport
  code; the browser follows the same profile contracts.

## 4. Acceptance criteria

For each supported profile, verify against its independently derived authoritative
interface that discovery selects the correct target or refuses ambiguity; explicit
selection cannot bypass type checks; unsupported controls remain unavailable; and camera
discovery displays all supported streams with visible failure/staleness states, and a
profile's sensors show current values, each bit of a status word, and visible
no-message/stale/last-value states. Verify
namespace behaviour only where the documented hardware interface supports it, including
refusal of overrides for other profiles. The installed console must run without sibling
source trees and use the same profile against matching physical and simulated wires.

For each mobile profile, verify key direction, the recorded key speeds and documented
limits, release/Space/input loss stop requests, ordinary-exit and handled-signal
(`SIGINT`/`SIGTERM`) cleanup, connection loss
ending teleop with a non-zero status, and a stop attempt at every start after validation
without resumed motion.
Check stop outcomes where documented feedback exists without making status recovery a
condition of re-enablement. Test an explicitly failed stop on a usable connection: commands
disable and require explicit re-enablement plus fresh input. Test lost teleop communication
without claiming a stop was delivered.

For the page, on ROS 1 and ROS 2 wires, verify that:
* nothing is sent before validation or on load, apart from the read-only joint-position
  poll and, for a teleoperable profile, the start-up stop after validation, and the page
  has no arm switch or `Esc` behaviour;
* dragging a topic-publish or action control streams throttled values and delivers the
  final value;
* on an action server that preempts, each goal preempts the previous one; on one that
  rejects a goal while another runs, or runs goals side by side, a drag keeps at most one
  goal outstanding, sends no cancel and delivers the final value after the running goal's
  result, and a rejected goal is shown as rejected; either way the goal connections stay
  bounded;
* service and action-group controls send once per click;
* out-of-range values are never sent, and pending and failed states are visible;
* release, focus loss, page hide or close, a target change and connection loss send no stop
  or cancel for a bounded control;
* for each mobile profile, base motion on the page passes the `teleop.sh` checks above (key
  direction, recorded speeds and documented limits, stop requests on release, `Space`, the
  Stop button, focus loss and page hide, a failed stop requiring explicit re-enablement
  plus fresh input, connection loss without claiming a stop), motion keys typed into an
  editable field move nothing, and an arm profile offers no base motion;
* connection loss disables the controls until reload;
* after a target change, values go only to the newly validated target.

Verify `python -m robot_console.fleet` passes on a matching wire and exits non-zero on
wrong types, missing required endpoints, ambiguous candidates (naming the candidates
found), a wire with no supported robot and an unreachable wire.
Verify that unknown robot ids are refused.
