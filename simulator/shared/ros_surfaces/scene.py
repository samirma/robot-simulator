"""The worktop's fixed camera rig, as a surface of its own.

The rig -- `overhead` and `side`, staged into the model by `tasks/apple_on_plate.py` --
watches the work surface. It is not a robot's: it would still be there with the arm
unbolted, and on real hardware it is a camera driver launched outside any robot's
namespace. It used to be rendered inside the SO-101's surface loop anyway, which meant a
kitchen with no arm in it -- `--robots ainex` -- compiled both cameras into the model and
published neither. Nothing was on the wire, nothing errored, and the live page showed a
lone head camera: the rig had been the arm's after all.

So the rig is a member of the fleet in its own right, attached under `SCENE_NAMESPACE`
by whichever engine staged the task, and rendered by the same `CameraStreams` every robot
uses. One publisher, one sequence counter, and the topics exist whenever the cameras do.
"""

from __future__ import annotations

import sys

# The two names are *defined* in the SO-101's module and imported here, not the other
# way round: the console's contract test reads that file by path with no package on its
# path and no MuJoCo installed, so the record of the rig's topics has to live somewhere
# that stays importable that way. This module is the rig's code; that one holds its
# contract. MuJoCo is imported inside the functions for the same reason.
from ros_surfaces.so101 import SCENE_CAMERA_TOPICS, SCENE_NAMESPACE  # noqa: F401


def probe_scene_cameras(model, cameras: dict[str, tuple[str, int, int]] | None = None
                        ) -> dict[str, tuple[str, int, int]]:
    """The subset of `cameras` whose MJCF camera actually exists in `model`.

    `CameraStreams` refuses a missing camera outright, and rightly -- a robot's own
    camera that is not there is a broken robot. The rig is different: a kitchen with no
    task staged has no rig, and that is a supported configuration, not an error. So the
    engine asks first and attaches nothing when there is nothing to render.
    """
    import mujoco

    wanted = SCENE_CAMERA_TOPICS if cameras is None else cameras
    return {
        topic: spec for topic, spec in wanted.items()
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, spec[0]) >= 0
    }


def attach_scene_rig(bus, *, model, cameras, jpeg_quality: int = 70, scene_option=None,
                     world_reset=None):
    """Put the rig on `bus` and return the per-step callback the fleet drives.

    Same closure shape as every robot surface: called with `MjData` each control period,
    with `None` to close the streams. `world_reset` is accepted because the fleet hands
    it to every member; the rig has no state a reset could restore.
    """
    from mujoco_bridge import CameraStreams

    streams = CameraStreams(model, dict(cameras), jpeg_quality, scene_option,
                            frame_of=bus.frame)
    print(f"scene rig under namespace {bus.ns}\n  pub {', '.join(streams.published)}",
          file=sys.stderr)

    def step(data):
        if data is None:
            streams.close()
            return
        streams.publish(bus, data, bus.next_seq(), float(data.time))

    return step
