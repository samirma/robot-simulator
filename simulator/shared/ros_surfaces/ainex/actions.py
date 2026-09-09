"""Action groups -- the AiNex's only grasping mechanism.

The real robot has no inverse-kinematics service and no Cartesian arm interface. Every
manipulation it does, including picking a block up, is a **recorded servo trajectory**
replayed open-loop: `MotionManager.run_action(name)` opens
`<action_path>/<name>.d6a` and steps through it frame by frame
(`ainex_kinematics/src/ainex_kinematics/motion_manager.py`). `/app/set_action` is how that
gets triggered over ROS. This module is the simulator's side of the same idea.

Two file formats, one runtime representation
--------------------------------------------
`.d6a` is the vendor's on-robot format: a SQLite database with a single `ActionGroup`
table whose rows are `(index, duration_ms, servo1 ... servo24)`.

**No `.d6a` files are shipped here.** They are vendor pose data with no stated licence,
and the ~50-file set that circulates publicly is a mirror of an SD-card image rather than
anything Hiwonder published. Reading the format anyway is the point: point `--action-dir`
at a real robot's `/home/ubuntu/software/ainex_controller/ActionGroups` and the genuine
motions play, so the licensed data stays an input the user supplies and never something
this repository redistributes.

The in-tree set is our own, deliberately small, and written as YAML because a SQLite blob
cannot be reviewed or diffed in a pull request. It is authored in **radians** rather than
servo counts, following the vendor's own `init_pose.yaml` -- a reader can check a pose
against that file directly. A user-supplied file shadows ours by name.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .servos import BY_ID, INIT_POSE, SERVOS, clamp, count_to_angle

#: Renamed from `actions/` when this moved into `shared/`: a data directory of that
#: name sat beside `actions.py` in one package and resolved only because a regular
#: module outranks a namespace package. It worked, and it was one import away from
#: not working.
ACTION_DIR = Path(__file__).resolve().parent / "action_groups"


@dataclass(frozen=True)
class ActionFrame:
    """One keyframe: hold these joint angles, then take `duration_s` to reach the next.

    `angles` always covers all 24 joints. The vendor's frames are dense by construction;
    the YAML ones are sparse and resolved on load (see `_load_yaml`).
    """

    duration_s: float
    angles: dict[str, float]


def load_action_dir(path: Path | None = None) -> dict[str, list[ActionFrame]]:
    """Read every `.yaml` and `.d6a` under `path`, keyed by file stem.

    A `.d6a` wins over a `.yaml` of the same name: if someone has pointed this at a real
    robot's action directory, they want that robot's motions, not our stand-ins.
    """
    directory = Path(path) if path is not None else ACTION_DIR
    actions: dict[str, list[ActionFrame]] = {}
    if not directory.is_dir():
        return actions

    for file in sorted(directory.glob("*.yaml")):
        actions[file.stem] = _load_yaml(file)
    for file in sorted(directory.glob("*.d6a")):
        actions[file.stem] = _load_d6a(file)
    return actions


def _load_d6a(path: Path) -> list[ActionFrame]:
    """Read Hiwonder's SQLite action-group format.

    Column layout is `motion_manager.run_action`'s: it skips two columns and then treats
    the rest positionally as servos 1..N, with the duration in milliseconds in column 1.
    Trailing columns beyond 24 -- some editors write spares -- are ignored rather than
    treated as an error, since the robot ignores them too.
    """
    frames: list[ActionFrame] = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        for row in db.execute("select * from ActionGroup"):
            counts = row[2 : 2 + len(BY_ID)]
            angles = {
                name: clamp(name, count_to_angle(name, count))
                for name, count in zip(BY_ID, counts)
            }
            frames.append(ActionFrame(duration_s=float(row[1]) / 1000.0, angles=angles))
    return frames


#: The torso's lean, radians about its lateral axis: the 25th channel an action group may
#: author beside the 24 servos. It is not a servo -- the real robot leans because its hip
#: chain tips the body over planted feet -- but this simulator's base rides the torso,
#: so the lean is a joint of its own (`ainex_model`, step 2) and a group that bends down
#: to pick something up has to say how far. Zero everywhere it is not mentioned.
BASE_PITCH = "base_pitch"


def _stance_lean() -> float:
    """The torso lean that puts the soles flat in the vendor's init pose.

    Lazily, like `_clamp_pitch` below: this module is otherwise pure, and the number is
    measured off the compiled model rather than typed -- see `ainex_model.stance_lean`.
    Cached because it costs a throwaway compile and never changes within a process.
    """
    global _STANCE_LEAN
    if _STANCE_LEAN is None:
        from ainex_model import build_spec, stance_lean  # noqa: PLC0415

        _STANCE_LEAN = stance_lean(build_spec())
    return _STANCE_LEAN


_STANCE_LEAN: float | None = None


#: Arms hanging at the sides with the hands pointing at the ground -- three joints per
#: arm, replacing the vendor's `init_pose.yaml` values for them.
#:
#: Applied over the vendor's numbers rather than edited into them: `servos.INIT_POSE` is a
#: verbatim transcription of that file, and the gait, the servo-count conversions and the
#: action groups all read it, so it stays exactly as shipped.
#:
#: Measured on the compiled model, hand position in the torso frame, and rendered from the
#: front and the side -- the numbers alone do not tell "down at the sides" from "folded
#: across the chest", and the first attempt at this passed the numbers and looked wrong.
#:
#: * **`sho_roll` decides up from down, and the vendor's sign raises the arm.** The URDF's
#:   zero has the arm straight out sideways, 254 mm from the torso; the vendor's `+1.293`
#:   swings it *up*, putting the hand 37.7 mm **above** the shoulder -- a goalpost, which
#:   is what this robot used to spawn in. `-pi/2` swings it down instead.
#: * **`el_yaw` is what held the forearm horizontal.** At the vendor's `-1.926` the
#:   forearm sits **113 degrees off straight-down**, which reads as elbows bent at a right
#:   angle. Sweeping it: `-1.5` gives 87.8 deg, `-1.0` gives 60.6 deg, `0.0` gives 10.8.
#: * **`el_pitch` is not what is left.** The residual 10.8 degrees is the elbow link's own
#:   offset and does not move with `el_pitch` -- the vendor's `-0.10` and zero both measure
#:   10.8 -- so zero is chosen for being the description's own value, not to null anything.
#:
#: Measured result: hands 74.0 mm out from the torso centreline and 180.4 mm below the
#: shoulder, against 78-93 mm out and 38-42 mm *above* it before.
#:
#: **Whether the vendor's `sho_roll` sign is wrong or ours is, is an open question.** A
#: sign decides up from down and nothing in the description says which way
#: `init_pose.yaml` means it. If it turns out `servo_controller.yaml` marks that pair
#: `min > max` -- the vendor's way of writing a servo mounted the other way round, which
#: `SERVOS` already records for the `sho_pitch` pair -- then it belongs in that table as
#: `flipped`, and only the two elbow entries here would remain.
RESTING_ARMS: dict[str, float] = {
    "l_sho_roll": -math.pi / 2, "r_sho_roll": math.pi / 2,
    "l_el_yaw": 0.0, "r_el_yaw": 0.0,
    "l_el_pitch": 0.0, "r_el_pitch": 0.0,
}


def rest_pose() -> dict[str, float]:
    """The pose the robot holds when nothing else drives it: the vendor's init pose with
    the arms down, and the torso leaned so that pose stands on its soles.

    The vendor's `init_pose` leg chain sums to -14.95 degrees rather than to zero, which
    on the real robot is `hip_pitch_offset` tipping the *body* forward over flat feet. Our
    torso is on planar joints, so with the lean at zero those 14.95 degrees landed on the
    feet: soles tilted toe-up, toes 37.6 mm off the surface, the robot balanced on two
    heel corners. Every gap check agreed it was standing perfectly, because they all
    measure the lowest single vertex and that vertex was the heel.

    A function rather than the `REST_POSE` constant it replaces, because the lean is
    measured off a compiled model and this module is imported long before one exists --
    and because a constant here would have to be built by a `dict()` copy at every call
    site anyway, which is what those sites already did.
    """
    return {**INIT_POSE, **RESTING_ARMS, BASE_PITCH: _stance_lean()}


def _clamp_pitch(value: float) -> float:
    # Lazily: this module is otherwise pure, and the range belongs to the model.
    from ainex_model import BASE_PITCH_RANGE  # noqa: PLC0415

    return float(min(max(value, BASE_PITCH_RANGE[0]), BASE_PITCH_RANGE[1]))


def _load_yaml(path: Path) -> list[ActionFrame]:
    """Read our own format: a list of frames, each a duration plus a sparse pose.

    Joints a frame does not mention **carry forward from the frame before**, starting from
    the vendor's init pose. That is what makes a grasp sequence readable: `clamp_left`
    says "close the left gripper" without having to restate the twenty-three joints that
    are holding the arm where the previous frame put it.

    A frame may also carry `base_pitch`, the torso's lean (see `BASE_PITCH`); it carries
    forward the same way and starts at zero.
    """
    import yaml  # noqa: PLC0415 -- only the YAML path needs it; .d6a users need not have it

    document = yaml.safe_load(path.read_text()) or {}
    current = rest_pose()
    frames: list[ActionFrame] = []

    for entry in document.get("frames", []):
        for name, angle in (entry.get("servos") or {}).items():
            if name not in SERVOS:
                raise ValueError(f"{path.name}: unknown joint {name!r}")
            current[name] = clamp(name, float(angle))
        if BASE_PITCH in entry:
            current[BASE_PITCH] = _clamp_pitch(float(entry[BASE_PITCH]))
        frames.append(
            ActionFrame(duration_s=float(entry.get("duration", 0.5)), angles=dict(current))
        )
    return frames


class ActionPlayer:
    """Steps a loaded action group at simulation time, returning the pose to hold.

    Frames are interpolated rather than stepped, because the real robot's servos move
    continuously between the counts an action group names -- `set_servos_position` gives
    them a duration to travel over, it does not teleport them. A stepped playback would
    also slam a position actuator to a new setpoint, which on a light limb rings.

    Not thread-safe, and does not need to be: it is only touched from the simulation
    thread. `ros_surface.py` hands it a name from a websocket thread and this starts on
    the next tick.
    """

    def __init__(self, frames: list[ActionFrame], start: dict[str, float]) -> None:
        self._frames = frames
        self._start = dict(start)
        self._index = 0
        self._elapsed = 0.0

    @property
    def finished(self) -> bool:
        return self._index >= len(self._frames)

    def step(self, dt: float) -> dict[str, float]:
        """Advance by `dt` and return the joint angles to hold now."""
        if self.finished:
            return dict(self._frames[-1].angles) if self._frames else dict(self._start)

        frame = self._frames[self._index]
        previous = self._frames[self._index - 1].angles if self._index else self._start

        self._elapsed += dt
        if frame.duration_s <= 0.0:
            alpha = 1.0
        else:
            alpha = min(self._elapsed / frame.duration_s, 1.0)

        rest = rest_pose()
        pose = {
            name: previous.get(name, rest[name]) * (1.0 - alpha) + target * alpha
            for name, target in frame.angles.items()
        }

        if alpha >= 1.0:
            self._index += 1
            self._elapsed = 0.0
        return pose
