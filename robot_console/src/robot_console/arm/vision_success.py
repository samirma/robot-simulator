"""Success inferred from the camera, not from a topic the simulator publishes.

The simulator used to answer the question directly on ``/task_success``. That is
privileged state: no camera sees it, a real SO-101 has no equivalent, and a policy
graded by it is graded by something outside the robot's own senses. This module
replaces it with a verdict computed from the overhead frame the policy is already
being handed, so the grader and the policy see the same world.

**What the camera can and cannot settle.** ``CONTRACT.md`` section 5 has five clauses.
Four of them survive the move intact:

* clause 1, apple within 0.080 m of the plate centre -- this is what the overhead view
  measures best, and it is measured *in the plate's own frame* rather than in pixels
  (see `plate_relative_radius`), so the camera's 62-degree tilt costs nothing;
* clause 3, at rest -- the apple's centroid stops moving between frames;
* clause 4, held for 1.0 s of simulated time -- the camera messages carry the same
  simulated stamps every other verdict in this package uses;
* clause 5, travelled from the spawn point -- measured against where the apple was first
  seen this episode rather than a hard-coded pixel, so it does not care which engine
  staged the scene or where the arm was mounted.

Clause 2, the apple's height, is the interesting one, because the obvious ways to get it
all fail and an indirect one works. Apparent size does not measure height here -- measured,
the on-plate apple images *smaller* than the one at spawn, since occlusion and shading move
that number far more than 20 mm of height does. The side camera looks like the answer, at
5.6 degrees below horizontal it is nearly a height sensor, and it is not: the arm occludes
the apple there routinely, and in one recorded success no apple blob is separable in the
side frame at all.

What does work is the projection itself. `read_frame` back-projects the apple's image
point onto the plane a *resting* apple would occupy, so an apple that is higher than that
lands short of its true position, and the error shrinks as it descends. A gripper lowering
the apple onto the plate therefore drags its projected position across the plane even
while the apple is barely moving horizontally -- 30 mm of sweep on a measured placement,
against 5.1 mm of worst-case jitter for an apple that has actually settled. `REST_RADIUS_M`
is set from that gap, and with it the verdict fires with the apple's true height inside the
contract's band on every placement tested. The height clause is enforced; it is just
enforced through the geometry of the camera rather than read off directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from robot_console.arm.ros_settings import (
    OVERHEAD_CAMERA_HEIGHT,
    OVERHEAD_CAMERA_NAME,
    OVERHEAD_CAMERA_WIDTH,
    SCENE_CAMERA_FOVY_DEG,
    SCENE_CAMERA_POSES,
    SCENE_CAMERA_XYAXES,
)

#: Physical plate radius, metres. The plate is the only object in frame whose true size
#: is known, so it is the ruler: everything else is measured as a fraction of it and then
#: converted. That is what makes the verdict independent of camera distance and of which
#: engine staged the scene.
PLATE_RADIUS_M = 0.10
#: Clause 1, metres. Expressed against the plate radius it is a normalised 0.80.
GATE_M = 0.080
#: Clause 3. An apple moving faster than this is passing through, not resting.
MAX_SPEED_MS = 0.01
#: Clause 5. How far the apple must have travelled from where it was first seen.
MIN_DISPLACEMENT_M = 0.25
#: Clause 4.
HOLD_SECONDS = 1.0

# -- detector thresholds, all measured on real frames from both engines ---------------
#
# The apple is not the only red thing on the table: the dressing includes a red bowl and
# a red mug, and a RoboCasa kitchen adds two red stove burners. Measured across frames
# from both engines, the four separate cleanly on area and on how solidly the blob fills
# its own bounding box -- the mug's red is a crescent, because it is a ring seen from
# above, and the bowl and burners are an order of magnitude larger:
#
#     apple    area  255- 293 px    extent 0.77-0.83   aspect 0.94-1.00
#     mug      area  350- 514 px    extent 0.39-0.47   aspect 1.06-1.32
#     bowl     area 2443-3482 px    extent 0.60-0.72
#     burners  area 1605-2294 px    (RoboCasa only)
APPLE_AREA_PX = (120, 800)
APPLE_MIN_EXTENT = 0.62
APPLE_ASPECT = (0.65, 1.5)
APPLE_HUE = 8          # red wraps zero: hue <= 8 or >= 172
APPLE_MIN_SAT = 110
APPLE_MIN_VAL = 60

#: The plate is white, so it is found by low saturation and high value. The trap is that
#: a MolmoSpaces counter is white marble and a RoboCasa kitchen has white cabinet fronts,
#: both of which are larger and brighter than the plate. Circularity rejects them: the
#: plate measures 0.65-0.83 across every frame tried, the marble 0.18, the cabinets
#: 0.11-0.14. The low end of the plate's range is a frame where the arm crosses it.
PLATE_MIN_AREA_PX = 1500
PLATE_MIN_CIRCULARITY = 0.55
PLATE_MAX_SAT = 40
PLATE_MIN_VAL = 150


#: The height above the work surface at which a resting apple's centre sits: the plate's
#: top face plus the apple's radius. Back-projection needs a plane to land on, and this is
#: the plane the predicate is asking about -- "is the apple sitting on the plate?" is the
#: same question as "does the apple's image ray meet z = 0.040 inside the plate?".
#:
#: Assuming it buys back part of what clause 2 lost. An apple the gripper is holding above
#: the plate is not on this plane, so its ray meets the plane displaced outward from where
#: the apple really is -- by roughly (height - 0.040) * tan(28 deg) for this camera, about
#: 53 mm for an apple held 100 mm too high. That pushes a held apple toward the gate rather
#: than through it. It is a partial recovery and not a height measurement: an apple held
#: only a few centimetres high still lands inside the gate, which is why the resting and
#: hold clauses do the rest of that work.
RESTING_CENTRE_Z = 0.040


class GroundProjector:
    """Turns image points into arm-frame coordinates on a chosen horizontal plane.

    The camera pose is known (`SCENE_CAMERA_POSES` / `SCENE_CAMERA_XYAXES`), so there is
    no reason to approximate perspective -- and approximating it was measurably wrong.
    MuJoCo's camera convention is the one this inverts: the camera looks down its own
    -z, with +x to image-right and +y to image-up.
    """

    def __init__(self, camera: str = OVERHEAD_CAMERA_NAME,
                 width: int = OVERHEAD_CAMERA_WIDTH, height: int = OVERHEAD_CAMERA_HEIGHT):
        xy = SCENE_CAMERA_XYAXES[camera]
        right = np.asarray(xy[:3], dtype=float)
        up = np.asarray(xy[3:], dtype=float)
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)
        # Third axis completes a right-handed frame; the camera looks along its negation.
        back = np.cross(right, up)
        self._rot = np.column_stack([right, up, back])   # camera -> world
        self._eye = np.asarray(SCENE_CAMERA_POSES[camera], dtype=float)
        fovy = math.radians(SCENE_CAMERA_FOVY_DEG[camera])
        self._fy = (height / 2.0) / math.tan(fovy / 2.0)
        self._fx = self._fy                              # square pixels
        self._cx, self._cy = width / 2.0, height / 2.0

    def project(self, x: float, y: float, z: float) -> tuple[float, float] | None:
        """Where an arm-frame point lands in the image, or None if it is behind."""
        cam = self._rot.T @ (np.array([x, y, z], dtype=float) - self._eye)
        if cam[2] >= -1e-9:                # the camera looks down its own -z
            return None
        return (self._cx + self._fx * (cam[0] / -cam[2]),
                self._cy - self._fy * (cam[1] / -cam[2]))

    def to_plane(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Where the ray through pixel ``(u, v)`` crosses the horizontal plane at ``z``."""
        ray_cam = np.array([(u - self._cx) / self._fx, -(v - self._cy) / self._fy, -1.0])
        ray = self._rot @ ray_cam
        if abs(ray[2]) < 1e-9:
            return None
        t = (z - self._eye[2]) / ray[2]
        if t <= 0:                    # the plane is behind the camera
            return None
        p = self._eye + t * ray
        return float(p[0]), float(p[1])


@dataclass(frozen=True)
class Blob:
    """A detected object in image coordinates."""

    x: float
    y: float
    area: float


@dataclass(frozen=True)
class Plate:
    """The plate as an ellipse, which is what a circle looks like from 62 degrees.

    Keeping the ellipse rather than collapsing it to a radius is what lets clause 1 be
    evaluated correctly: a fixed pixel radius would be too strict across the minor axis
    and too generous along the major one.
    """

    cx: float
    cy: float
    semi_major: float
    semi_minor: float
    angle_deg: float

    def relative_radius(self, x: float, y: float) -> float:
        """Where a point sits in plate radii: 0 at the centre, 1.0 on the rim."""
        t = math.radians(self.angle_deg)
        dx, dy = x - self.cx, y - self.cy
        u = (dx * math.cos(t) + dy * math.sin(t)) / max(self.semi_major, 1e-6)
        v = (-dx * math.sin(t) + dy * math.cos(t)) / max(self.semi_minor, 1e-6)
        return math.hypot(u, v)

    @property
    def metres_per_pixel(self) -> float:
        """Scale from the plate's known radius, along its unforeshortened axis."""
        return PLATE_RADIUS_M / max(self.semi_major, 1e-6)


@dataclass(frozen=True)
class Reading:
    """One frame's worth of evidence. ``None`` where the frame did not show enough."""

    apple: Blob | None
    plate: Plate | None
    #: The apple's position on the resting plane, in arm-frame metres.
    apple_xy: tuple[float, float] | None
    #: The plate centre on the same plane, in arm-frame metres. Read from the image
    #: rather than taken from the task's constants, so a re-staged plate is followed
    #: and the verdict never depends on knowing where the scene put things.
    plate_xy: tuple[float, float] | None
    #: Apple-to-plate-centre distance in metres, the quantity clause 1 gates on.
    distance_m: float | None
    inside_gate: bool

    @property
    def usable(self) -> bool:
        return self.apple_xy is not None and self.plate_xy is not None


def _mask(bgr: np.ndarray, *, red: bool) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if red:
        m = ((h <= APPLE_HUE) | (h >= 180 - APPLE_HUE)) & (s > APPLE_MIN_SAT) & (v > APPLE_MIN_VAL)
        return cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = (s < PLATE_MAX_SAT) & (v > PLATE_MIN_VAL)
    m = m.astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def find_apple(bgr: np.ndarray) -> Blob | None:
    """Find the apple, or None if nothing in frame looks like one.

    Returns the *largest* qualifying blob rather than the first, so a stray speck of red
    on a fixture cannot outrank the apple.
    """
    n, _, stats, cent = cv2.connectedComponentsWithStats(_mask(bgr, red=True), 8)
    best: Blob | None = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (APPLE_AREA_PX[0] <= area <= APPLE_AREA_PX[1]) or w == 0 or h == 0:
            continue
        if area / float(w * h) < APPLE_MIN_EXTENT:
            continue
        if not (APPLE_ASPECT[0] <= w / h <= APPLE_ASPECT[1]):
            continue
        if best is None or area > best.area:
            best = Blob(float(cent[i][0]), float(cent[i][1]), float(area))
    return best


def find_plate(bgr: np.ndarray) -> Plate | None:
    """Find the plate as an ellipse, or None if no circular white region is in frame."""
    contours, _ = cv2.findContours(_mask(bgr, red=False), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best: Plate | None = None
    best_circ = PLATE_MIN_CIRCULARITY
    for c in contours:
        area = cv2.contourArea(c)
        if area < PLATE_MIN_AREA_PX or len(c) < 5:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        circ = area / (math.pi * r * r) if r > 0 else 0.0
        if circ < best_circ:
            continue
        (cx, cy), (d1, d2), ang = cv2.fitEllipse(c)
        best_circ = circ
        best = Plate(cx, cy, max(d1, d2) / 2.0, min(d1, d2) / 2.0, ang)
    return best


_PROJECTOR: GroundProjector | None = None


def _projector() -> GroundProjector:
    global _PROJECTOR
    if _PROJECTOR is None:
        _PROJECTOR = GroundProjector()
    return _PROJECTOR


def read_frame(bgr: np.ndarray) -> Reading:
    """Locate the apple and the plate in one overhead frame and measure the gap.

    Both are back-projected onto the plane a resting apple's centre occupies, so the
    distance comes out in metres in the arm's own frame and can be compared directly
    with the contract's 0.080 m -- no pixel gate, and no dependence on how far away the
    camera happens to be.
    """
    apple, plate = find_apple(bgr), find_plate(bgr)
    if apple is None or plate is None:
        return Reading(apple, plate, None, None, None, False)
    proj = _projector()
    apple_xy = proj.to_plane(apple.x, apple.y, RESTING_CENTRE_Z)
    plate_xy = proj.to_plane(plate.cx, plate.cy, RESTING_CENTRE_Z)
    if apple_xy is None or plate_xy is None:
        return Reading(apple, plate, None, None, None, False)
    d = math.dist(apple_xy, plate_xy)
    return Reading(
        apple=apple,
        plate=plate,
        apple_xy=apple_xy,
        plate_xy=plate_xy,
        distance_m=d,
        inside_gate=d < GATE_M,
    )


# -- the held verdict, across frames --------------------------------------------------
#
# Clause 3 asks for the apple to be moving slower than 0.01 m/s, and vision cannot answer
# that question as asked. Measured against the geometry it replaces, a single frame places
# the apple to about 16 mm; at the camera's ~8 Hz an apple travelling at exactly the
# contract's limit moves 1.25 mm between frames, so the quantity being tested is an order
# of magnitude below the noise on the instrument. Reporting a speed here would be
# inventing precision.
#
# What vision can say is that the apple has not gone anywhere, which is what clause 3 is
# for -- it exists to reject an apple rolling *through* the plate region rather than
# settling in it. So clauses 3 and 4 are enforced together and geometrically: the apple
# must stay inside the gate *and* within `REST_RADIUS_M` of where it was, continuously,
# for `HOLD_SECONDS` of simulated time.
#
# This radius also turns out to be what recovers the height clause, which is why it is
# small. Because `read_frame` back-projects onto the *resting* plane, an apple that is
# higher than that lands short of where it really is, by an amount that shrinks as it
# descends -- so a gripper lowering the apple onto the plate drags its projected position
# across the plane even while the apple is barely moving horizontally. Measured on a
# placement: the apple falling from z=0.085 to z=0.039 swept its projected position
# 30 mm, while a genuinely settled apple jitters 0.7 mm frame to frame and 5.1 mm at
# worst. Those are six to forty times apart, so the descent is easy to reject.
#
# The threshold was swept against that data rather than picked. At 20 mm the verdict
# fired with the apple still 80 mm up and travelling at 40 mm/s -- a placement called
# four seconds before it happened. At 8 mm all three placements fired at z = 0.040 to
# 0.045, inside the contract's 0.040 +/- 0.015 band, with the apple's true speed at
# 0.000 m/s. 6 mm is no better and sits close to the 5.1 mm noise floor, so 8 mm it is:
# a shade over 1.5x the worst jitter, and well under a quarter of the descent sweep.
REST_RADIUS_M = 0.008


@dataclass(frozen=True)
class Verdict:
    """The contract predicate as the camera can answer it."""

    reading: Reading
    #: The instantaneous half: inside the gate, travelled from spawn, and not drifting.
    #: Recorded per step so the offline scorer can re-derive the hold without a network.
    placed: bool
    #: `placed` sustained for the full hold.
    held: bool
    #: How long the instantaneous verdict has been continuously true, simulated seconds.
    hold_elapsed: float
    #: Clause 5: how far the apple has moved from where this episode first saw it.
    displacement_m: float | None
    #: The plate centre this verdict used, in arm-frame metres.
    plate_xy: tuple[float, float] | None


class VisionTracker:
    """Turns a stream of overhead frames into the contract's held verdict.

    One instance per episode; `reset` between them.

    The plate is estimated once and then *held*, as a running median over every frame
    seen. It is a static object, so re-deciding where it is on every frame only lets the
    arm crossing it drag the answer around -- measured, per-frame estimates wander by up
    to 31 mm and the median settles to 14 mm of the true centre, which on an 80 mm gate
    is the difference between six missed placements and none.
    """

    def __init__(self, hold_seconds: float = HOLD_SECONDS) -> None:
        self.hold_seconds = float(hold_seconds)
        self.reset()

    def reset(self) -> None:
        self._plates: list[tuple[float, float]] = []
        self._spawn: tuple[float, float] | None = None
        self._hold_since: float | None = None
        self._anchor: tuple[float, float] | None = None
        self._elapsed = 0.0

    @property
    def plate_xy(self) -> tuple[float, float] | None:
        if not self._plates:
            return None
        arr = np.asarray(self._plates)
        return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))

    def update(self, bgr: np.ndarray, stamp: float | None) -> Verdict:
        """Fold one frame in and return the verdict as it stands."""
        reading = read_frame(bgr)
        if reading.plate_xy is not None:
            self._plates.append(reading.plate_xy)
        plate = self.plate_xy
        apple = reading.apple_xy

        if apple is None or plate is None:
            # A frame that shows nothing is not evidence against the hold, but it is not
            # evidence for it either: the hold needs continuity, so it restarts.
            self._hold_since = self._anchor = None
            self._elapsed = 0.0
            return Verdict(reading, False, False, 0.0, None, plate)

        if self._spawn is None:
            self._spawn = apple
        displacement = math.dist(apple, self._spawn)

        inside = math.dist(apple, plate) < GATE_M
        travelled = displacement > MIN_DISPLACEMENT_M
        steady = self._anchor is not None and math.dist(apple, self._anchor) <= REST_RADIUS_M

        # Decided before the branch below touches `_anchor`: evaluating it afterwards
        # reads the anchor this very frame just set, and the first genuinely placed frame
        # then reports itself as not placed.
        placed = bool(inside and travelled and (self._anchor is None or steady))

        if placed:
            if self._hold_since is None or stamp is None:
                self._hold_since = stamp
                self._anchor = apple
            self._elapsed = 0.0 if stamp is None or self._hold_since is None else max(
                0.0, stamp - self._hold_since)
        else:
            self._hold_since = self._anchor = None
            self._elapsed = 0.0

        return Verdict(
            reading=reading,
            placed=placed,
            held=self._elapsed >= self.hold_seconds,
            hold_elapsed=round(self._elapsed, 4),
            displacement_m=displacement,
            plate_xy=plate,
        )
