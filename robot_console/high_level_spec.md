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
camera types (a camera of an optional add-on, such as a wrist camera, or one that the
authoritative boot does not start, is itself optional; a camera row whose topic, or whose
`sensor_msgs/CameraInfo` topic, has no topic row declares that output topic itself, so one more
camera is one more row),
the read-only `sensors` the view page may show (manufacturer topics, each an output topic of
the profile with its type, the message fields to show and a stale threshold, or none for a
latched topic; §2.5), the per-axis teleoperation speed a held key commands (for mobile profiles),
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
  camera streams (§2.3), every profile camera subscribed once the target is validated. Cameras paused on congestion (§2.3) are resumed with `C`. Focus loss is that window losing keyboard focus, input loss is the
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
  subscribes to nothing but the target's cameras and its profile's sensors, and it offers no generic topic publication or
  subscription. A subscription is not a command: it is never a "send" in the sense below.
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
validated, they list every discovered supported raw camera stream, clearly marked as not tied
to a target; since no profile says what such a stream costs the link, none is streamed by
itself: the page previews one on request, one at a time, at most 2 frames per second, and
teleop.sh shows them off. Accept the documented ROS 1 or ROS 2 raw image types and encodings
supported by the profile, and `sensor_msgs/CompressedImage` streams the profile lists (its
encodings are the codecs it documents, jpeg or png; the browser decodes them, teleop.sh with
pygame), without substituting a simulated image source. Each stream shows live images and a visible unavailable, unsupported,
stale or failed state when applicable. Profiles define a finite stale-image threshold
from documented rates or an explicitly identified console policy. Never present a frozen
frame as live. Every camera in the profile is subscribed once the target is validated, with
nothing waiting behind a click: the user chose seeing every stream over link economy (on the
ROSMASTER X3 PLUS too, whose profile therefore sets no link budget). Each card can stop its
stream and start it again. Having no cameras is a visible state, not a connection failure for
independent controls. A camera the profile marks optional that is not on the wire is shown
as unavailable and does not fail validation. A profile lists only the streams the page
shows. A depth camera's depth stream (a raw single-channel 16-bit image, `16UC1`) is a
camera of the profile of kind `depth`, beside the camera's colour stream. The profile
gives its metres per value and its range (near, far); the stream is drawn as a
false-colour map scaled to that range (nearer, warm, to farther, cool; values beyond the
range take the end colours), a value of 0 (no depth) is black, and the page shows a
legend with the range and the black "no depth" swatch. A depth camera documents `16UC1`
only; another encoding is an unsupported frame. An infrared camera's stream (a raw
single-channel image, such as `mono16`) is a camera of kind `ir`: each frame is drawn in grey
stretched from its own darkest value (black) to its own brightest (white), all black for a
flat frame, with a note saying so; it documents single-channel encodings only. Both entry
points and the wire check decode depth and infrared frames this way, and a 16-bit stream
with neither (an untied one) is stretched from 0 to its own maximum. A camera may name its
documented `sensor_msgs/CameraInfo` topic, which the page shows with the stream (it does not
subscribe to it). An optional camera may carry a note, shown with its missing state and in
the wire check, saying what starts it (for example a launch other than the authoritative
boot). A camera whose topic is advertised but may never deliver (a driver that keeps the
topic when it runs without that stream) may carry a note that is shown, as a neutral
"no frames" state rather than stale, while no frame has arrived; the camera is then optional.
A camera may carry a short caption shown on its card (for example that a depth stream is
registered to colour while its CameraInfo carries the parameters after that registration).
Exiting closes subscriptions and the connection.

Frames share the one rosbridge connection that carries base motion and its stop, as base64
JSON (a 640x480 rgb8 frame is about 1.2 MB). The console bounds what they take, shows it, and
stops them when the link backs up:

* Every subscription (camera or sensor) asks rosbridge for `queue_length` 1 with its
  `throttle_rate`, so rosbridge keeps only the newest message between sends
  (rosbridge_suite 0.11.17 `rosbridge_library/internal/subscription_modifiers.py#L105-L111`).
  On ROS 1 the standard launch runs the Autobahn/Twisted server
  (`rosbridge_server/launch/rosbridge_websocket.launch#L35`, `#L59`;
  `rosbridge_server/scripts/rosbridge_websocket#L42-L49`), whose `OutgoingValve` holds
  rosbridge's next message until Twisted has written the previous one out of its own buffer
  (`rosbridge_server/src/rosbridge_server/autobahn_websocket.py#L114-L151`, the wait at
  `#L136-L137`; registered at `#L181-L182`), so the newest-only queue does apply. What has been
  written still waits in Twisted's buffer, the kernel's socket buffer and the Wi-Fi's queues;
  none of those is bounded by the console, so on a link slower than the streams a frame's delay
  can grow to seconds. The protection against that is the console's own: the rate caps and
  budget below, and pausing on congestion.
* Every camera subscription asks for at most the profile camera's `max_rate_hz` frames per
  second, or 10 when it records none (console policy). A profile that needs a lower cap records
  it with its reason: the wheeled bases and the AiNex, whose only stop shares the link and which
  have no watchdog, cap their cameras at 2 frames/s. The one exception is a compressed stream
  (`sensor_msgs/CompressedImage`), which such a profile may cap at up to 5 frames/s, with that
  reason, when its nominal load at the cap stays below a raw 640x480 rgb8 stream's at 2 frames/s
  (the ROSMASTER X3 PLUS arm camera's JPEG: about 4.9 against 19.7 Mbit/s; each frame holds the
  link about a tenth as long, so the stop waits behind less). For every camera the frame period at its cap
  must be at most half its stale threshold (one missing frame is not "no frame") and at most 1 s
  (so the live bound below stays short).
* A profile may set a link budget (`link_budget_mbps`, console policy) for its camera streams
  together. A stream's load is its message size (measured, or before its first frame a 640x480
  frame of its first encoding as base64 JSON, a compressed one taken as a tenth of rgb8) times
  its rate cap. All its cameras, which start on validation, must fit the budget together
  (checked when the profile loads), so nothing on by default is ever refused. Starting a
  stopped stream that would take the cameras over the budget is refused, with the numbers. A
  profile without a budget says why (the ROSMASTER X3 PLUS: the user's choice above). A refusal stays on its card until the stream is started, or the cameras are
  paused or resumed (it then no longer describes the link). The page shows the link's measured incoming load and, where there is a budget,
  the cameras' share of it, in warning colours from 80 % of it.
* Congestion, judged the same way by the page and teleop.sh: R1 a camera is LAGGING while the
  newest time answer (below) took at least 0.3 s (a LAGGING camera with quick answers is not the
  link: they share it), once its frames span over 2.4 s and show no steady offset (below); R2 a time answer took more than 1 s (answered within the last 5 s; an
  answer later than 3 s still counts); R3 a time question went unanswered for 3 s, now or given up
  within the last 3 s, while frames arrived within the last second (shown yet or not). The
  warning names the rule that fired, with its numbers. On congestion every camera stream (and
  every throttled sensor, such as a point cloud, §2.5) is unsubscribed and the page says so prominently, as a warning
  (never a neutral state), at the top of the cameras and where the operator drives, until the
  operator resumes them; teleop.sh does the same, resumed with C.
* A frame is only stored when it arrives and is decoded in a later task, the newest only, so
  key, button and focus events (and the stop they request) are handled between frames and never
  wait behind images; base motion and its stop are sent on the socket at once, never queued
  behind a frame.

Link health: a camera is live only while the image it shows is current, judged again on every
refresh of the page or window, not only when the frame arrived. A frame that arrived more than
1 s after its stamp (robot time; console policy) is LAGGING. A shown image is live while its age
is at most 1 s plus 1.5 frame periods at the stream's cap (the card shows that bound; 1.75 s at
2 frames/s, 1.15 s at 10), so a steady lag below 1 s never flickers between frames; past it, with
no newer frame, the image is stale. No image older than that bound reads live, whatever the
robot's clock does: a decoded frame is shown only once a time answer (below) has read the robot's
clock after the frame was taken (its question sent after the frame arrived, or the time it read
not before the frame's stamp); until then the image on screen keeps ageing and the clock is asked
at once. A shown image's age is bounded with the lowest and highest offset of every time answer
from 10 s before the frame arrived until now, so a clock step before, while or after the frame was
taken only widens the bounds (the card shows the image's age and how uncertain it is). The one
case left is a frame taken at a clock reading no answer in that window saw: the robot's clock
stepped and stepped back between two answers (asked at least every 2 s), or a frame delayed more
than 10 s whose clock stepped back by about that delay more than 10 s before it arrived; a frame
whose bounds were confirmed by a reading that may predate a backward step is asked about again
before it is called LAGGING. The camera's stale threshold bounds how long it waits for a frame
at all. Robot time comes from an independent clock reference: the console asks rosapi's
`/rosapi/get_time` (rosapi runs beside rosbridge in the standard launch, rosbridge_suite 0.11.17
`rosbridge_server/launch/rosbridge_websocket.launch#L82-L86`; it answers `rospy.get_rostime()`,
`rosapi/scripts/rosapi_node#L177-L178`; it is ROS infrastructure, never part of a profile) about
every 2 s, read-only. A question sent at console time t0 and answered at t1 with robot time T
bounds the clock offset to [T - t1, T - t0], whatever the delays on the way, so a delayed answer
(a backed-up link) only widens its own bounds; the console's estimate (for the steady-offset
check) is the intersection of the recent answers' bounds, and an answer whose bounds exclude its
midpoint is taken as a clock step and replaces the older answers. Console time is monotonic, so
a step of the console's wall clock changes nothing. A frame stamped more than 0.25 s after its own
arrival means the robot's clock is inconsistent: its age is unknown. On ROS 1 the console first
reads `/use_sim_time` through rosapi, again until it is known (a question unanswered for 3 s
counts for R3); a robot on simulated time has no reference, but the console keeps asking the time
and timing the answers, so R2 and R3 work as on any robot. A stream whose frames sit at a steady
offset (over 3 s, within 0.25 s, beyond 2 s, while the newest time answer took at most 1 s and
less than half that offset, an unknown round trip not counting) cannot be told apart from stamps
on another clock (the Astra SDK stamps on its host clock), so it is shown LAGGING, never live,
with that note; R1 skips it, since a queue the time answers share would delay them as much. When there is no usable reference (rosapi's
time absent or not answering within 3 s, simulated time or not yet known, a robot clock reading
0, a reference older than 10 s, a frame without a stamp, or bounds too wide to decide) the age is
unknown: the stream is shown as "lag unknown", a warning, and never as live; once its newest
frame arrived longer ago than the live bound it is stale. A card shows a frame's age only while
it judges the image on screen against the clock, never on a stopped or paused stream. The wire check (§2.4) reports delivery only and
does not measure lag.

### 2.4 Wire check — `python -m robot_console.fleet`

```sh
python -m robot_console.fleet [--url ws://…] [--expect <id>]
```

* Discover the robots on the wire through typed signatures and check each against its
  profile: every required endpoint with its type, and nothing else under its names but the
  profile's optional rows and ROS infrastructure (workspace spec §1.3). `--expect` narrows
  this to one robot id, which must present its whole typed interface. Every ambiguous
  case fails, naming the candidates found.
* Report each profile camera as live, stale, silent or missing (§2.3). The streams are checked
  one at a time, so the check never takes more of the link than one stream: each is subscribed at its rate cap only until it has delivered the two frames that show it
  live (or the check's window ends), then unsubscribed before the next. The check publishes nothing
  and ends with a non-zero status when validation fails, an expected camera is not live, no
  supported robot is discovered, or the wire is unreachable. A missing or not live optional
  camera is reported as optional and does not fail the check.

### 2.5 Sensor panel

The view page shows the selected target's profile `sensors`: manufacturer data topics
(motor and joint states, arm status, battery, IMU, odometry, lidar, point clouds) that the
profile records as output topics, each with its type, the message fields shown and a stale
threshold, or none for a latched or non-periodic topic, and may carry a short caption shown on
its card (for example why no unit is shown). A field is a path into the message
shown as a number (with its unit and decimals), text, a boolean, a list, a status word (the
documented meaning of each bit), a vector (x, y, z with its unit), an orientation (a
quaternion shown as roll, pitch and yaw in degrees; a zero quaternion shows that the message
carries no orientation; it may also be shown re-expressed in another frame through a documented
fixed mount, labelled with that frame) or the number of elements of an array.

* A unit is shown only where the profile's sources establish it.
* A number field may be the profile's headline: it is also shown large, with its state,
  above the cameras (a robot's battery voltage, say). A number field may carry a documented
  level below which it is marked low, in its card and its headline (a battery's alarm voltage).
* A number field may list documented values that mean "no reading" (a value a driver reports
  before its hardware has answered, or when it did not answer), each with what it means: such
  a value is shown as that meaning, never as a reading, and is never marked low.
* A sensor may record, as a console policy, a minimum period between the messages the page
  asks rosbridge for (`throttle_s`, shorter than its stale threshold); the sensor says so.
  Otherwise the page asks for at most ten messages per second. rosbridge still receives every
  message from ROS, so this bounds only the websocket. A throttled sensor (a large message,
  such as a point cloud subscribed at one message per 5 s) has a Stop/Start button on its card,
  is shown stopped while the user has stopped it, and pauses and resumes with the cameras on
  congestion (§2.3).
* A sensor whose publisher sends only on an event (an ordinary, not latched, publisher, such
  as a command echo or a mode toggle) is marked `event_only`: until an event comes it shows a
  neutral "event-only, none yet" state, then its last event, never stale. The panel's summary
  counts the periodic and latched sensors as "N of M reporting", and states the event-only
  ones (how many seen) apart, so waiting for them never reads as a failure.
* A `sensor_msgs/LaserScan` sensor may be drawn as a top-down plot of its returns (forward
  up, rings for scale), with only the returns within its range limits drawn and the nearest
  one named with its bearing. Its radius follows the bulk of the returns (their 90th
  percentile, rounded up to 1, 2 or 5 times a power of ten), so a few far returns do not shrink
  it; returns beyond it are drawn on the rim and counted; the user can zoom in and out and
  return to the automatic radius.
* Float values that are not finite and reach the page as bare `NaN`/`Infinity` tokens (as
  Python's json writes them) are read as no value, not as an unreadable message; string
  contents are never changed.

* The panel is read-only: the page subscribes to every sensor topic of the profile once the
  target is validated, with nothing waiting behind a click, and never publishes, calls or cancels anything
  for them. The page offers in-page links to its cameras, sensors, model and controls, so the
  sensors are reachable on a narrow screen. A profile without
  sensors shows no panel.
* Each sensor shows its resolved topic, its current field values, its message count and
  recent message rate (as delivered to the page), the age of its last message and a state:
  no message yet, live, stale (the last message is older than the threshold; the values are
  marked frozen), last value (a latched topic that has delivered, never stale), stopped (by the
  user) or paused (on congestion), missing from the wire, or disconnected. Values are never
  presented as current once the state is not live or last value. A status word shows each documented bit
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
discovery displays all supported streams with visible failure/staleness states (depth as its
false-colour map, infrared stretched in grey, an absent optional camera with its note, a
silent one with its note), each subscription at its rate cap, every camera on without a click and
stopped and started again on request, and full-size (640x480) frames decoded; and a
profile's sensors show current values in their units, each bit of a status word, a headline
value and a value below its documented level marked low, a documented no-reading value shown
as such, a scan plot of a LaserScan, every sensor (point clouds too) reporting without a click,
and visible no-message/stale/last-value states. Verify
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
