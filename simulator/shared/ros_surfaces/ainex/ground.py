"""Keep the AiNex's soles on whatever is under them, and let it fall when nothing is.

The torso rides five position-actuated joints (`ainex_model`, step 2). x, y and yaw are
what the gait drives. z is nobody's to command: this module solves it every control
tick, so that the stance sole sits exactly on the surface a ray-cast finds beneath it --
a worktop, an uneven counter, the floor -- for whatever the legs and the torso lean are
doing. A crouch lowers the body because the legs fold and the sole stays put; the gait's
swing foot lifts because the stance foot, not the higher one, is the one on the ground.

When no foot finds a surface within reach -- the robot has walked off the edge of the
worktop -- the setpoint falls: `vz` integrates g, the target drops by `vz*dt`, and the
robot lands the tick its sole would cross the surface the ray finds below. It is a fall
of the *setpoint*, integrated here, not gravity acting on the torso: `gravcomp` stays at
1 on every body and the limbs hold their pose on the way down, exactly as they do on a
real robot that has stopped balancing. The point is that a robot which leaves its
surface arrives on the one below it, instead of hanging in the air where it left.

Pure with respect to ROS: takes a model, data and a prefix, and writes one actuator.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mujoco
import numpy as np

from mujoco_bridge import is_loose

#: The bodies whose soles touch the ground. `ainex_model.FEET` is the same tuple and is
#: the one the model measures its stance from; it cannot be imported at module scope here
#: (see the note in `__init__` -- that import is a circle), so the two are held equal by
#: an assertion where the lazy import already happens.
FEET = ("l_ank_roll_link", "r_ank_roll_link")
#: How far above the sole the probe ray starts. It has to start above: a ray that starts
#: inside a box reports that box's far face, and a sole sitting on a counter starts
#: inside the counter's collider by however deep the servo let it sink.
PROBE_M = 0.10
#: A sole this far above its surface still counts as standing on it. Wider than the
#: gait's step height so a swing foot is "supported" too -- harmless, because the foot
#: with the *highest* surface relative to its sole is the one that sets the height, and
#: that is the stance foot -- and wide enough to ride the bumps on an iTHOR counter.
SUPPORT_GAP_M = 0.05
GRAVITY = 9.81
#: How far past an own-geom hit the probe restarts. It has to be smaller than the gap
#: between the sole's lowest geom and the surface it stands on -- 26 um at the standing
#: equilibrium -- because a step over that face restarts the ray *inside* the worktop and
#: `mj_ray` then reports the box's underside, exactly the trap PROBE_M is written against.
#: At 1 mm, which is what this was, a standing robot read its surface 100 mm below its
#: sole on every tick a foot geom poked above it -- so a gap wider than SUPPORT_GAP_M, so
#: "walked off the edge", so a fall of 67 mm and four ticks climbing back: a permanent
#: 2 Hz, 69 mm bob that reads as a robot juddering and not as a ray-cast.
RECAST_STEP_M = 1e-6
#: Enough re-casts to pass through the robot's own foot and shin -- which the probe starts
#: above and therefore inside -- **and** through whatever is lying on the floor under it,
#: since a loose geom is skipped the same way. Unbounded re-casting would turn a bad pose
#: into a hang, so this is measured rather than generous-by-feel: sweeping 62 500 downward
#: columns over FloorPlan1, 94% reach ground in one cast and the worst needs **37**,
#: because an iTHOR object is a decomposed hull -- 1080 loose geoms across 37 objects,
#: some thirty convex pieces each -- and every piece in the column costs a cast. The
#: robot's own feet add 3 standing and 5 walking. 64 covers that with headroom; a budget
#: of 12 did not, and the give-up warning below is what said so, from the island edge of
#: a real kitchen.
MAX_RECAST = 64


@dataclass(frozen=True)
class GroundState:
    supported: bool
    falling: bool
    #: sole height above the surface under it, for the lowest foot; +inf with no surface
    gap: float
    surface_z: float | None


class GroundFollow:
    def __init__(self, model, prefix: str = "", *, probe: float = PROBE_M,
                 support_gap: float = SUPPORT_GAP_M, g: float = GRAVITY) -> None:
        self._model = model
        self._probe = probe
        self._support_gap = support_gap
        self._g = g
        joint = model.joint(f"{prefix}base_z")
        self._z_adr = int(model.jnt_qposadr[joint.id])
        self._z_act = int(model.actuator(f"{prefix}base_z_act").id)
        # The robot's own bodies, so the probe passes through its foot and shin rather
        # than reporting them as the ground.
        self._own = {
            b for b in range(model.nbody)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(prefix)
        } if prefix else set()
        # Imported here, not at the top: `ainex_model` imports this package's servo table,
        # and the package's `__init__` imports the surface, which imports this module --
        # a top-level import here is a circle that fails at the first `import ainex_model`.
        from ainex_model import FEET as MODEL_FEET, LowestPoint  # noqa: PLC0415

        # Ground-following a different pair of bodies than the model measured its stance
        # from is a robot standing at the wrong height and nothing saying so.
        assert tuple(FEET) == tuple(MODEL_FEET), (FEET, MODEL_FEET)

        self._feet = LowestPoint(model, prefix, bodies=tuple(f"{prefix}{f}" for f in FEET))
        self._vz = 0.0
        self._falling = False
        self._geomid = np.zeros(1, dtype=np.int32)
        self._warned_recast = False

    def reset(self) -> None:
        self._vz = 0.0
        self._falling = False

    @property
    def falling(self) -> bool:
        return self._falling

    def _surface_below(self, data, point: np.ndarray) -> float | None:
        """World z of the ground straight below `point`, or None.

        Ground is what the world is made of, not what is lying on it. The robot's own
        geoms are skipped because the probe starts inside its foot, and **loose** geoms --
        `mujoco_bridge.is_loose`, anything on a free joint -- are skipped because an apple
        is not a floor. Without that second clause the robot climbs: measured on the task's
        20 mm apple, the moment a sole's probe point crossed it the surface came back as
        0.5392 against the worktop's 0.5000 and the robot was lifted 39 mm onto its crown,
        while the apple itself did not move by a millimetre. It reads as a robot levitating.
        """
        start = np.ascontiguousarray(point + [0.0, 0.0, self._probe], dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0])
        travelled = 0.0
        for _ in range(MAX_RECAST + 1):
            dist = mujoco.mj_ray(self._model, data, start, vec, None, 1, -1, self._geomid)
            if self._geomid[0] < 0 or dist < 0.0:
                return None
            total = travelled + dist
            body = int(self._model.geom_bodyid[self._geomid[0]])
            if body in self._own or is_loose(self._model, body):
                travelled = total + RECAST_STEP_M
                start = np.ascontiguousarray(
                    point + [0.0, 0.0, self._probe - travelled], dtype=np.float64
                )
                continue
            return float(point[2] + self._probe - total)
        # Out of casts, still inside the robot or a pile of loose objects. `step` reads
        # None as "nothing below", so the robot is about to be told the floor is z = 0 and
        # dropped there. Say so once: a silent None is why the 1 mm step above was
        # invisible for as long as it was.
        if not self._warned_recast:
            self._warned_recast = True
            print(f"ainex: ground probe gave up after {MAX_RECAST + 1} casts without "
                  f"reaching ground -- treating the surface as z = 0", file=sys.stderr)
        return None

    def step(self, data, dt: float) -> GroundState:
        z = float(data.qpos[self._z_adr])
        # Per foot: how far its sole would have to move to sit on its own surface.
        corrections: list[float] = []
        gaps: list[float] = []
        surfaces: list[float] = []
        for sole in self._feet.per_body(data).values():
            surface = self._surface_below(data, sole)
            if surface is None:
                # Nothing under this foot at all: the world's floor is z = 0 by both
                # engines' convention, and a robot has to land somewhere.
                surface = 0.0
            gap = float(sole[2] - surface)
            gaps.append(gap)
            surfaces.append(surface)
            if -self._probe <= gap <= self._support_gap:
                corrections.append(surface - float(sole[2]))

        if corrections:
            # Standing. The foot with the largest correction is the one whose surface
            # is highest relative to its sole -- the stance foot -- and it wins, so no
            # supported sole is ever left below its surface.
            self._vz = 0.0
            self._falling = False
            target = z + max(corrections)
            data.ctrl[self._z_act] = target
            return GroundState(True, False, min(gaps), max(surfaces))

        # Falling. Land the tick the drop would carry a sole through its surface.
        self._falling = True
        self._vz += self._g * dt
        drop = self._vz * dt
        gap = min(gaps)
        if drop >= gap:
            data.ctrl[self._z_act] = z - gap
            self._vz = 0.0
            self._falling = False
            return GroundState(True, False, 0.0, surfaces[int(np.argmin(gaps))])
        data.ctrl[self._z_act] = z - drop
        return GroundState(False, True, gap, surfaces[int(np.argmin(gaps))])
