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

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .servos import BY_ID, INIT_POSE, SERVOS, clamp, count_to_angle

ACTION_DIR = Path(__file__).resolve().parent / "actions"


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


def _load_yaml(path: Path) -> list[ActionFrame]:
    """Read our own format: a list of frames, each a duration plus a sparse pose.

    Joints a frame does not mention **carry forward from the frame before**, starting from
    the vendor's init pose. That is what makes a grasp sequence readable: `clamp_left`
    says "close the left gripper" without having to restate the twenty-three joints that
    are holding the arm where the previous frame put it.
    """
    import yaml  # noqa: PLC0415 -- only the YAML path needs it; .d6a users need not have it

    document = yaml.safe_load(path.read_text()) or {}
    current = dict(INIT_POSE)
    frames: list[ActionFrame] = []

    for entry in document.get("frames", []):
        for name, angle in (entry.get("servos") or {}).items():
            if name not in SERVOS:
                raise ValueError(f"{path.name}: unknown joint {name!r}")
            current[name] = clamp(name, float(angle))
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

        pose = {
            name: previous.get(name, INIT_POSE[name]) * (1.0 - alpha) + target * alpha
            for name, target in frame.angles.items()
        }

        if alpha >= 1.0:
            self._index += 1
            self._elapsed = 0.0
        return pose
