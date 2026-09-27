"""The camera verdict: apple-on-plate graded from what the rig and the arm publish.

Nothing on the wire answers the task's question. This module answers it from public
observations only -- the observations a policy could have as well:

* synchronized ``/scene/overhead`` and ``/scene/side`` colour frames,
* both cameras' ``camera_info`` (intrinsics),
* the rig mount poses (`ros_settings.SCENE_CAMERAS`, duplicated exactly from the
  simulator task's constants -- calibration, not episode state),
* ``/joint_states``, matched to the frames by stamp.

**Apple pose.** The apple is segmented in both views and its centroid triangulated from
the two calibrated rays, so its height is measured rather than assumed -- the overhead
view alone cannot tell an apple on the plate from one held still above it. Speed comes
from timestamped poses `SPEED_BASELINE_S` apart, never from image stillness.

**Release.** Forward kinematics from ``/joint_states`` puts both fingers in the same rig
frame (``kinematics.finger_segments``). Every finger must stay `FINGER_CLEARANCE_M` from
the apple centre through the whole hold, so an apple held motionless over the plate
fails.

**Decision.** An episode passes when, for at least `HOLD_SECONDS` at its end, every
synchronized sample shows the apple within `MAX_HORIZONTAL_DIST_M` of the plate centre
horizontally, within `Z_TOLERANCE_M` of `RESTING_Z_M`, at no more than `MAX_SPEED_MPS`,
and released. A sample with missing synchronization, calibration, segmentation or joint
state, or a gap in the hold longer than `MAX_SAMPLE_GAP_S`, fails the episode: the scorer
fails closed rather than guessing.

The plate centre is located in the overhead view (a static white ellipse) and
back-projected onto the plate's top plane; the median over the graded frames is used,
because the arm crossing the plate drags any single frame's estimate around.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from robot_console.arm.kinematics import ARM_JOINTS, GRIPPER_JOINT, finger_segments, \
    point_segment_distance
from robot_console.arm.ros_settings import (
    OVERHEAD_CAMERA_NAME,
    SCENE_CAMERAS,
    SIDE_CAMERA_NAME,
)

# ------------------------------------------------------------------ the pass criterion
#
# The console's own spec (§2.3) states these; the simulator task's constants carry the
# same numbers, and `tests/arm/test_vision_success.py` duplicates them rather than
# importing them, so a change here is a change someone has to make twice.

#: Apple centre to plate centre, horizontally, metres.
MAX_HORIZONTAL_DIST_M = 0.080
#: Where a resting apple's centre sits: plate top plus apple radius, metres.
RESTING_Z_M = 0.040
#: How far from `RESTING_Z_M` the apple centre may be, metres.
Z_TOLERANCE_M = 0.015
#: At rest, not passing through, m/s.
MAX_SPEED_MPS = 0.01
#: How long, at the episode's end, every clause must hold, seconds of stamp time.
HOLD_SECONDS = 1.0
#: Every finger this far from the apple centre, metres: released, not held.
FINGER_CLEARANCE_M = 0.05

# ------------------------------------------------------------------ measurement limits

#: Two frames, or a frame and a joint state, count as one instant when their stamps are
#: within this. The rig renders both views in one job with one stamp, so they normally
#: match exactly; half the rig's 10 Hz period is the most a pair may disagree by.
SYNC_TOLERANCE_S = 0.05
#: The longest gap between consecutive samples inside the hold. The rig runs at 10 Hz;
#: a gap past two and a half frames is a hole in the evidence, not a slow camera.
MAX_SAMPLE_GAP_S = 0.25
#: Speed is |p(t) - p(t - d)| / d over at least this baseline. A pose triangulated from
#: 640x480 JPEGs jitters by about a millimetre; over one 0.1 s frame that is the whole
#: 0.01 m/s budget, over half a second it is a fifth of it.
SPEED_BASELINE_S = 0.5
#: A triangulated apple must reproject into both views within this many pixels of the
#: segmented centroids, or the two blobs were not the same object.
MAX_REPROJECTION_PX = 4.0
#: The apple's radius, metres, for the apparent-size sanity check on a triangulation.
APPLE_RADIUS_M = 0.020
#: Observed apparent radius over the one a whole apple would have at the triangulated
#: range. Below the lower bound the blob is a fragment (an apple mostly occluded, whose
#: centroid is not its centre); above the upper, it is not the apple.
APPARENT_SIZE_RATIO = (0.45, 1.6)
#: The plate's top face, metres: the plane its outline is back-projected onto.
PLATE_TOP_Z_M = 0.0204

# -- segmentation thresholds, measured on both engines' frames ----------------------
#
# The apple is the only saturated red, roughly round, compact blob of its size; the
# dressing's red mug is a crescent (extent 0.39-0.47) and bowls and burners are an
# order of magnitude larger. Area bounds are generous because the side camera sees the
# apple smaller than the overhead one, and partial occlusion shrinks it further.
APPLE_HUE = 8          # red wraps zero: hue <= 8 or >= 172
APPLE_MIN_SAT = 110
APPLE_MIN_VAL = 60
APPLE_AREA_PX = (25, 2500)
APPLE_MIN_EXTENT = 0.55
APPLE_ASPECT = (0.5, 2.0)

#: The plate is white: low saturation, high value, circular. Measured: the plate runs
#: sat 0-5 against a marble worktop's 15-243; circularity rejects white cabinets.
PLATE_MIN_AREA_PX = 1500
PLATE_MIN_CIRCULARITY = 0.55
PLATE_MAX_SAT = 10
PLATE_MIN_VAL = 150
#: Fallback when the flat mask merges plate and worktop: the brightest slice of the
#: white field (the plate is the brightest white thing on both engines).
PLATE_BRIGHT_PERCENTILE = 95

REQUIRED_JOINTS: tuple[str, ...] = (*ARM_JOINTS, GRIPPER_JOINT)


# ------------------------------------------------------------------ calibration


@dataclass(frozen=True)
class CameraModel:
    """A calibrated pinhole: intrinsics from ``camera_info``, pose from the rig mount.

    The optical frame is ROS's: x image-right, y image-down, z forward. The mount pose
    is MuJoCo ``xyaxes`` (image-right, image-up) in the arm base frame, so the optical
    axes are (right, -up, -(right x up)).
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    rotation: np.ndarray   # optical -> base
    eye: np.ndarray

    @classmethod
    def from_camera_info(cls, name: str, info: Mapping[str, Any] | None) -> CameraModel:
        """Raise ``ValueError`` on anything missing or inconsistent with the mount."""
        if name not in SCENE_CAMERAS:
            raise ValueError(f"no mount pose for camera {name!r}")
        if not isinstance(info, Mapping):
            raise ValueError(f"no camera_info for {name}")
        k = info.get("k", info.get("K"))
        try:
            k = [float(v) for v in k]
            width, height = int(info["width"]), int(info["height"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"malformed camera_info for {name}: {exc}") from exc
        if len(k) != 9 or not all(math.isfinite(v) for v in k) or k[0] <= 0 or k[4] <= 0:
            raise ValueError(f"camera_info for {name} carries no usable intrinsics")
        pos, xyaxes, _fovy, resolution = SCENE_CAMERAS[name]
        if (width, height) != tuple(resolution):
            raise ValueError(f"camera_info for {name} is {width}x{height}, the rig's "
                             f"calibration is {resolution[0]}x{resolution[1]}")
        right = np.asarray(xyaxes[:3], dtype=float)
        up = np.asarray(xyaxes[3:], dtype=float)
        right /= np.linalg.norm(right)
        up -= right * float(right @ up)
        up /= np.linalg.norm(up)
        back = np.cross(right, up)
        rotation = np.column_stack([right, -up, -back])
        return cls(k[0], k[4], k[2], k[5], width, height, rotation,
                   np.asarray(pos, dtype=float))

    def ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction, in the base frame, through pixel ``(u, v)``."""
        d = self.rotation @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
        return d / np.linalg.norm(d)

    def project(self, point: Sequence[float]) -> tuple[float, float] | None:
        cam = self.rotation.T @ (np.asarray(point, dtype=float) - self.eye)
        if cam[2] <= 1e-9:
            return None
        return (self.cx + self.fx * cam[0] / cam[2], self.cy + self.fy * cam[1] / cam[2])

    def to_plane(self, u: float, v: float, z: float) -> tuple[float, float] | None:
        """Where the ray through ``(u, v)`` meets the horizontal plane at ``z``."""
        d = self.ray(u, v)
        if abs(d[2]) < 1e-9:
            return None
        t = (z - self.eye[2]) / d[2]
        if t <= 0:
            return None
        p = self.eye + t * d
        return float(p[0]), float(p[1])


# ------------------------------------------------------------------ segmentation


@dataclass(frozen=True)
class Blob:
    x: float
    y: float
    area: float

    @property
    def radius_px(self) -> float:
        return math.sqrt(max(self.area, 0.0) / math.pi)


def decode(data: Any) -> np.ndarray | None:
    """JPEG/PNG bytes (or rosbridge's base64 string) to a BGR image, or None."""
    if data is None:
        return None
    if isinstance(data, str):
        try:
            data = base64.b64decode(data, validate=True)
        except (ValueError, TypeError):
            return None
    try:
        buf = np.frombuffer(bytes(data), dtype=np.uint8)
    except TypeError:
        return None
    if buf.size == 0:
        return None
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return image if image is not None and image.ndim == 3 else None


def apple_candidates(bgr: np.ndarray) -> list[Blob]:
    """Every red blob in frame the right shape to be the apple, largest first.

    Red wraps hue zero, and the two sides of the wrap are segmented separately as well as
    together: measured on the side view, the apple resting on the plate reads hue 0-7
    while the red bowl it touches in that projection reads 172-179, so one "red" mask
    merges them into a single 77x35 blob that is neither, while the split masks separate
    them. The union still matters -- the overhead view shows the lit apple straddling the
    wrap. Triangulation picks the pair that is one object; see `locate_apple`.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (s > APPLE_MIN_SAT) & (v > APPLE_MIN_VAL)
    low, high = (h <= APPLE_HUE) & vivid, (h >= 180 - APPLE_HUE) & vivid
    out: list[Blob] = []
    for red in (low | high, low, high):
        mask = cv2.morphologyEx(red.astype(np.uint8) * 255, cv2.MORPH_OPEN,
                                np.ones((3, 3), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w, hgt = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
            if not (APPLE_AREA_PX[0] <= area <= APPLE_AREA_PX[1]) or w == 0 or hgt == 0:
                continue
            if area / float(w * hgt) < APPLE_MIN_EXTENT:
                continue
            if not (APPLE_ASPECT[0] <= w / hgt <= APPLE_ASPECT[1]):
                continue
            blob = Blob(float(cent[i][0]), float(cent[i][1]), float(area))
            if not any(abs(b.x - blob.x) < 0.5 and abs(b.y - blob.y) < 0.5 for b in out):
                out.append(blob)
    return sorted(out, key=lambda b: -b.area)


def _plate_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_circ = None, PLATE_MIN_CIRCULARITY
    for c in contours:
        area = cv2.contourArea(c)
        if area < PLATE_MIN_AREA_PX or len(c) < 5:
            continue
        (_, _), r = cv2.minEnclosingCircle(c)
        circ = area / (math.pi * r * r) if r > 0 else 0.0
        if circ < best_circ:
            continue
        (cx, cy), _, _ = cv2.fitEllipse(c)
        best, best_circ = (float(cx), float(cy)), circ
    return best


def find_plate(bgr: np.ndarray) -> tuple[float, float] | None:
    """The plate's image centre in an overhead frame, or None."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    white = (s < PLATE_MAX_SAT) & (v > PLATE_MIN_VAL)

    def clean(m: np.ndarray) -> np.ndarray:
        m = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    found = _plate_from_mask(clean(white))
    if found is not None or int(white.sum()) < PLATE_MIN_AREA_PX:
        return found
    threshold = np.percentile(v[white], PLATE_BRIGHT_PERCENTILE)
    return _plate_from_mask(clean((v >= threshold) & white))


# ------------------------------------------------------------------ triangulation


@dataclass(frozen=True)
class Triangulation:
    position: np.ndarray
    reprojection_px: float


def triangulate(a: CameraModel, pa: tuple[float, float],
                b: CameraModel, pb: tuple[float, float]) -> Triangulation | None:
    """Midpoint of the two rays' closest approach, and its worst reprojection error."""
    da, db = a.ray(*pa), b.ray(*pb)
    w0 = a.eye - b.eye
    aa, bb, ab = float(da @ da), float(db @ db), float(da @ db)
    denom = aa * bb - ab * ab
    if denom < 1e-12:
        return None
    sa = (ab * float(db @ w0) - bb * float(da @ w0)) / denom
    sb = (aa * float(db @ w0) - ab * float(da @ w0)) / denom
    if sa <= 0 or sb <= 0:
        return None
    point = 0.5 * ((a.eye + sa * da) + (b.eye + sb * db))
    errors = []
    for cam, pixel in ((a, pa), (b, pb)):
        proj = cam.project(point)
        if proj is None:
            return None
        errors.append(math.dist(proj, pixel))
    return Triangulation(point, max(errors))


def _size_ratio(cam: CameraModel, point: np.ndarray, blob: Blob) -> float:
    expected = cam.fy * APPLE_RADIUS_M / max(float(np.linalg.norm(point - cam.eye)), 1e-6)
    return blob.radius_px / expected if expected > 0 else math.inf


def locate_apple(overhead: CameraModel, over_bgr: np.ndarray,
                 side: CameraModel, side_bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Triangulate the apple from one synchronized pair: (position, why-not)."""
    top, low = apple_candidates(over_bgr), apple_candidates(side_bgr)
    if not top:
        return None, "apple not segmented in the overhead view"
    if not low:
        return None, "apple not segmented in the side view"
    best: Triangulation | None = None
    for a in top:
        for b in low:
            t = triangulate(overhead, (a.x, a.y), side, (b.x, b.y))
            if t is None or t.reprojection_px > MAX_REPROJECTION_PX:
                continue
            ratios = (_size_ratio(overhead, t.position, a), _size_ratio(side, t.position, b))
            if not all(APPARENT_SIZE_RATIO[0] <= r <= APPARENT_SIZE_RATIO[1] for r in ratios):
                continue
            if best is None or t.reprojection_px < best.reprojection_px:
                best = t
    if best is None:
        return None, "no red blob in the two views triangulates to one apple"
    return best.position, ""


# ------------------------------------------------------------------ recorded samples
#
# The embodiment records one dict per synchronized rig pair it received (see
# `embodiment.SO101RosEmbodiment`), in each step's ``info["rig_samples"]``:
#
#     {"overhead": {"stamp": s, "data": <jpeg bytes>},
#      "side": {"stamp": s, "data": <jpeg bytes>} | None,
#      "camera_info": {"overhead": {...}, "side": {...}},
#      "joint_state": {"stamp": s, "name": [...], "position": [...]} | None}

RIG_SAMPLES_KEY = "rig_samples"


def _stamp(part: Any) -> float | None:
    if not isinstance(part, Mapping):
        return None
    try:
        value = float(part.get("stamp"))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


@dataclass
class Measurement:
    stamp: float
    apple: np.ndarray | None = None
    plate_px: tuple[float, float] | None = None
    clearance_m: float | None = None
    problem: str = ""

    @property
    def ok(self) -> bool:
        return not self.problem and self.apple is not None and self.clearance_m is not None


def measure(sample: Mapping[str, Any]) -> Measurement:
    """Everything one synchronized sample says, or the first reason it says nothing."""
    stamp = _stamp(sample.get("overhead"))
    m = Measurement(stamp=stamp if stamp is not None else math.nan)
    if stamp is None:
        m.problem = "overhead frame has no stamp"
        return m
    infos = sample.get("camera_info") or {}
    # The plate is static, so any calibrated overhead frame locates it, synchronized
    # with anything else or not.
    over = decode(sample["overhead"].get("data"))
    try:
        over_cam = CameraModel.from_camera_info(OVERHEAD_CAMERA_NAME, infos.get(OVERHEAD_CAMERA_NAME))
    except ValueError:
        over_cam = None
    if over is not None and over_cam is not None and \
            over.shape[:2] == (over_cam.height, over_cam.width):
        m.plate_px = find_plate(over)
    side_stamp = _stamp(sample.get("side"))
    if side_stamp is None:
        m.problem = "no side frame"
        return m
    if abs(side_stamp - stamp) > SYNC_TOLERANCE_S:
        m.problem = f"overhead and side frames {abs(side_stamp - stamp):.3f} s apart"
        return m
    try:
        over_cam = CameraModel.from_camera_info(OVERHEAD_CAMERA_NAME, infos.get(OVERHEAD_CAMERA_NAME))
        side_cam = CameraModel.from_camera_info(SIDE_CAMERA_NAME, infos.get(SIDE_CAMERA_NAME))
    except ValueError as exc:
        m.problem = f"calibration: {exc}"
        return m
    side = decode(sample["side"].get("data"))
    if over is None or side is None:
        m.problem = "frame missing or undecodable"
        return m
    if over.shape[:2] != (over_cam.height, over_cam.width) or \
            side.shape[:2] != (side_cam.height, side_cam.width):
        m.problem = "frame size does not match its camera_info"
        return m
    joints = sample.get("joint_state")
    joint_stamp = _stamp(joints)
    if joint_stamp is None:
        m.problem = "no joint state"
        return m
    if abs(joint_stamp - stamp) > SYNC_TOLERANCE_S:
        m.problem = f"joint state {abs(joint_stamp - stamp):.3f} s from the frames"
        return m
    try:
        by_name = dict(zip(joints["name"], (float(p) for p in joints["position"]), strict=True))
        values = [by_name[name] for name in REQUIRED_JOINTS]
    except (KeyError, TypeError, ValueError):
        m.problem = "joint state lacks the arm's joints"
        return m
    apple, why = locate_apple(over_cam, over, side_cam, side)
    if apple is None:
        m.problem = why
        return m
    m.apple = apple
    fingers = finger_segments(values[:5], values[5])
    m.clearance_m = min(point_segment_distance(apple, a, b) for a, b in fingers)
    return m


def plate_centre(measurements: Sequence[Measurement],
                 camera_info: Mapping[str, Any] | None) -> tuple[float, float] | None:
    """Median plate centre on its top plane over every frame that showed it."""
    try:
        cam = CameraModel.from_camera_info(OVERHEAD_CAMERA_NAME, (camera_info or {}).get(
            OVERHEAD_CAMERA_NAME))
    except ValueError:
        return None
    points = [cam.to_plane(u, v, PLATE_TOP_Z_M) for u, v in
              (m.plate_px for m in measurements if m.plate_px is not None)]
    points = [p for p in points if p is not None]
    if not points:
        return None
    arr = np.asarray(points)
    return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))


@dataclass
class Assessment:
    passed: bool
    reason: str
    hold_s: float = 0.0
    final_position: list[float] | None = None
    plate_xy: list[float] | None = None
    horizontal_m: float | None = None
    height_error_m: float | None = None
    speed_mps: float | None = None
    clearance_m: float | None = None
    samples: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {k: v for k, v in self.__dict__.items() if k != "extra"}
        out.update(self.extra)
        return out


def _order(samples: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: dict[float, Mapping[str, Any]] = {}
    for sample in samples:
        stamp = _stamp(sample.get("overhead")) if isinstance(sample, Mapping) else None
        if stamp is not None:
            seen[stamp] = sample
    return [seen[s] for s in sorted(seen)]


def assess(samples: Sequence[Mapping[str, Any]]) -> Assessment:
    """Apply the pass criterion to an episode's synchronized samples, failing closed."""
    ordered = _order(samples)
    if not ordered:
        return Assessment(False, "no synchronized rig observations recorded")
    stamps = [float(_stamp(s["overhead"])) for s in ordered]  # type: ignore[arg-type]
    end = stamps[-1]
    start = end - HOLD_SECONDS
    anchors = [i for i, t in enumerate(stamps) if t <= start + 1e-9]
    if not anchors:
        return Assessment(False, f"observations cover only {end - stamps[0]:.2f} s, "
                                 f"the hold needs {HOLD_SECONDS:g} s", samples=len(ordered))
    first = anchors[-1]
    # Poses from before the hold are needed for the speed at its start.
    earliest = start - SPEED_BASELINE_S - 2 * MAX_SAMPLE_GAP_S
    lo = next((i for i, t in enumerate(stamps) if t >= earliest), first)
    lo = min(lo, first)
    measured = {i: measure(ordered[i]) for i in range(lo, len(ordered))}

    plate = plate_centre(list(measured.values()), ordered[-1].get("camera_info"))
    result = Assessment(False, "", samples=len(ordered))
    if plate is not None:
        result.plate_xy = [round(plate[0], 4), round(plate[1], 4)]
    final = measured[len(ordered) - 1]
    if final.apple is not None:
        result.final_position = [round(float(v), 4) for v in final.apple]

    def fail(reason: str, at: float) -> Assessment:
        result.reason = f"{reason} at t={at:.2f} s ({end - at:.2f} s before the end)"
        return result

    if plate is None:
        if final.problem:
            return fail(final.problem, stamps[-1])
        result.reason = "plate not found in the overhead view"
        return result
    for i in range(first + 1, len(ordered)):
        gap = stamps[i] - stamps[i - 1]
        if gap > MAX_SAMPLE_GAP_S:
            return fail(f"{gap:.2f} s without a synchronized sample", stamps[i])

    worst_speed = worst_h = worst_z = 0.0
    min_clear = math.inf
    for i in range(first, len(ordered)):
        m = measured[i]
        if not m.ok:
            return fail(m.problem or "no measurement", stamps[i])
        assert m.apple is not None and m.clearance_m is not None
        h = math.dist((float(m.apple[0]), float(m.apple[1])), plate)
        dz = abs(float(m.apple[2]) - RESTING_Z_M)
        worst_h, worst_z = max(worst_h, h), max(worst_z, dz)
        min_clear = min(min_clear, m.clearance_m)
        if m.clearance_m < FINGER_CLEARANCE_M:
            return fail(f"a finger {m.clearance_m:.3f} m from the apple centre, not released "
                        f"(needs >= {FINGER_CLEARANCE_M:g})", stamps[i])
        if h > MAX_HORIZONTAL_DIST_M:
            return fail(f"apple {h:.3f} m from the plate centre "
                        f"(gate {MAX_HORIZONTAL_DIST_M:g})", stamps[i])
        if dz > Z_TOLERANCE_M:
            return fail(f"apple centre at z={float(m.apple[2]):.3f} m, not resting at "
                        f"{RESTING_Z_M:g}+/-{Z_TOLERANCE_M:g}", stamps[i])
        base = [j for j in range(lo, i) if stamps[i] - stamps[j] >= SPEED_BASELINE_S]
        if not base:
            return fail("no pose early enough to measure speed", stamps[i])
        j = base[-1]
        for k in range(j, i):
            if not measured[k].ok:
                return fail(f"speed unmeasurable: {measured[k].problem or 'no pose'}", stamps[k])
            if stamps[k + 1] - stamps[k] > MAX_SAMPLE_GAP_S:
                return fail("speed unmeasurable: gap in the poses", stamps[k + 1])
        speed = float(np.linalg.norm(m.apple - measured[j].apple)) / (stamps[i] - stamps[j])
        worst_speed = max(worst_speed, speed)
        if speed > MAX_SPEED_MPS:
            return fail(f"apple moving at {speed:.4f} m/s (limit {MAX_SPEED_MPS:g})", stamps[i])

    result.passed = True
    result.hold_s = round(end - stamps[first], 3)
    result.horizontal_m = round(worst_h, 4)
    result.height_error_m = round(worst_z, 4)
    result.speed_mps = round(worst_speed, 5)
    result.clearance_m = round(min_clear, 4)
    result.reason = (f"held {result.hold_s:.2f} s: apple within {worst_h:.3f} m of the plate "
                     f"centre, {worst_z:.3f} m of resting height, <= {worst_speed:.4f} m/s, "
                     f"fingers >= {min_clear:.3f} m away")
    return result
