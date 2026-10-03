"""MuJoCo helpers the simulation's modules share, one copy each (spec §4 "Shared logic
exists once"): a heading as a quaternion and a rotation, a body's subtree, the world box
of every geom, and removing an element from a spec on either engine's MuJoCo (3.5 and 3.3).

The worktop survey (`worktop_survey.py`) keeps its own ports of the reference project's
helpers, line for line, so the robot stands exactly where the reference stood it.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np


def yaw_quat(yaw: float) -> list[float]:
    """(w, x, y, z) of a rotation by `yaw` about +z."""
    return [float(math.cos(yaw / 2)), 0.0, 0.0, float(math.sin(yaw / 2))]


def yaw_matrix(yaw: float) -> np.ndarray:
    """The 3x3 rotation by `yaw` about +z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def subtree(model, root: int) -> set:
    """The ids of body `root` and every body inside it (none for a negative id). MuJoCo
    numbers a body after its parent, so one pass in id order finds them."""
    if root < 0:
        return set()
    inside = np.zeros(model.nbody, bool)
    inside[root] = True
    for b in range(root + 1, model.nbody):
        inside[b] = inside[model.body_parentid[b]]
    return {int(b) for b in np.nonzero(inside)[0]}


def geom_aabbs(model, data):
    """World AABB (lo, hi) of every geom; planes get an infinite horizontal extent."""
    R = data.geom_xmat.reshape(-1, 3, 3)
    c = data.geom_xpos + np.einsum("nij,nj->ni", R, model.geom_aabb[:, :3])
    e = np.einsum("nij,nj->ni", np.abs(R), model.geom_aabb[:, 3:])
    lo, hi = c - e, c + e
    planes = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
    lo[planes, :2], hi[planes, :2] = -np.inf, np.inf
    lo[planes, 2] = hi[planes, 2] = data.geom_xpos[planes, 2]
    return lo, hi


def spec_delete(spec, element) -> None:
    """Remove an element from a spec: MuJoCo 3.5 through `spec.delete`, 3.3 through
    `detach_body` for a body and the element's own `delete` otherwise."""
    if hasattr(spec, "delete"):
        spec.delete(element)
    elif isinstance(element, mujoco.MjsBody) and hasattr(spec, "detach_body"):
        spec.detach_body(element)
    elif hasattr(element, "delete"):
        element.delete()
