"""The one simulation: a scene, the robots spawned into it, and its physics loop.

The world owns the MuJoCo spec, model and data. A robot is added or removed by editing
the spec and recompiling with `MjSpec.recompile`, which carries simulation time and the
positions, velocities, actuator activations and controls of every element that survives
into the new model -- scene objects and the other robots keep moving as they were, and
no controller or watchdog state is reinitialised (the wire-side controllers never notice).

A worktop robot comes with its staging (`worktop_objects.Staging`): the six objects it
brings, and the scene's loose objects cleared from its working area. Both enter in the
robot's own recompile and leave in the recompile that removes it; a cleared object comes
back with the position and velocity it had when it was cleared.

Physics runs on its own thread against the wall clock. Everything that reads or writes
the model or data holds `world.lock`.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

import robot_model
from robot_model import RobotModel

STEP_BATCH = 3          # physics steps per call between serving other threads
RTF_WINDOW_S = 10.0
RTF_WARN = 0.90


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


@dataclass
class Instance:
    """A robot in the world (pending while its spawn starts up, then running)."""

    id: str
    rm: RobotModel
    prefix: str
    placement: str
    xyz: np.ndarray
    yaw: float
    support_geoms: tuple = ()
    ports: tuple = ()
    state: str = "pending"
    token: str = ""
    frame: object = None
    body: object = None
    lease: object = None                 # the spawn's connection
    wires: list = field(default_factory=list)
    info: dict = field(default_factory=dict)
    #: the objects staged with a worktop robot (`worktop_objects.Staging`), or None
    staging: object = None


class Mirror:
    """A private MjData that reproduces the world's geometry from a snapshot, so ray casts
    and scene updates for rendering run on their own thread, not the physics one."""

    def __init__(self):
        self.version = -1
        self.model = None
        self.data = None

    def update(self, snap: dict):
        if snap["version"] != self.version:
            self.model = snap["model"]
            self.data = mujoco.MjData(self.model)
            self.version = snap["version"]
        d = self.data
        d.qpos[:] = snap["qpos"]
        d.mocap_pos[:] = snap["mocap_pos"]
        d.mocap_quat[:] = snap["mocap_quat"]
        d.time = snap["time"]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)
        return self.model, d


class World:
    def __init__(self, scene, log=None):
        self.scene = scene
        self.spec = scene.spec
        self.lock = threading.RLock()
        self.log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        #: bumped at every recompile: renderers and bindings rebuild on a change.
        self.version = 0
        self.robots: dict[str, Instance] = {}
        #: the six worktop objects, part of the scene itself once staged (`stage_scene`)
        self.scene_staging = None
        self.scene_nbody = self.model.nbody
        self.scene_ngeom = self.model.ngeom
        self._stop = threading.Event()
        self._thread = None
        self.rtf = 1.0
        self.listeners = []   # callables(world) run after a recompile, under the lock
        self._calls = queue.SimpleQueue()
        # actuator targets from the wires, by full actuator name, applied by the physics
        # thread before its next step (see set_ctrl)
        self._ctrl_lock = threading.Lock()
        self._ctrl_pending: dict = {}

    # ------------------------------------------------------------------ robots

    def add(self, inst: Instance, staging=None) -> None:
        """Attach the robot at its pose -- with its staging (`worktop_objects.Staging`:
        objects added, scene objects cleared) if it has one -- and recompile once, keeping
        every other state. On failure both are rolled back and the world is unchanged."""
        with self.lock:
            rm = inst.rm
            spec = rm.spec.copy() if hasattr(rm.spec, "copy") else rm.spec
            self._contact_bits(spec, inst)
            top = spec.body(rm.root)
            staged = False
            displaced = None
            body = None
            try:
                if staging is not None:
                    staging.capture(self.model, self.data)
                    staged = True
                    if self.scene_staging is not None:
                        # this arm stands elsewhere than the scene's own objects: they
                        # move to it for as long as it is here
                        displaced = self.scene_staging
                        displaced.remove_objects(self.spec)
                    staging.apply(self.spec)
                frame = self.spec.worldbody.add_frame(pos=[float(v) for v in inst.xyz],
                                                      quat=yaw_quat(inst.yaw))
                body = frame.attach_body(top, inst.prefix, "")
                inst.frame, inst.body = frame, body
                model, data = self.spec.recompile(self.model, self.data)
            except Exception:
                if body is not None:
                    robot_model._delete(self.spec, body)
                self._drop_prefixed(inst.prefix)
                if staged:
                    staging.undo(self.spec)
                if displaced is not None:
                    displaced.add_objects(self.spec)
                inst.frame = inst.body = None
                raise
            if displaced is not None:
                staging.place_objects(model, data)
                inst.info["displaced_scene_objects"] = True
            self._pose_new(model, data, inst)
            mujoco.mj_forward(model, data)
            inst.staging = staging
            self._swap(model, data)
            self.robots[inst.id] = inst

    def stage_scene(self, staging) -> None:
        """Make the worktop objects part of the scene (spec §2.2): staged once, at start,
        whatever robots come later, and never taken out. Counted in the scene's own body
        and geom totals."""
        with self.lock:
            staging.capture(self.model, self.data)
            staging.apply(self.spec)
            try:
                model, data = self.spec.recompile(self.model, self.data)
            except Exception:
                staging.undo(self.spec)
                raise
            mujoco.mj_forward(model, data)
            self.scene_staging = staging
            self.scene_nbody = model.nbody
            self.scene_ngeom = model.ngeom
            self._swap(model, data)

    ROBOT_BITS = [1 << k for k in range(1, 16)]

    def _contact_bits(self, spec, inst: Instance) -> None:
        """Robots collide with each other as they do with the scene. Each robot gets a
        contact bit of its own: its collision geoms add it to their contype and every
        other robot's bit to their conaffinity, so two robots always form a contact pair
        while a robot's own geoms keep exactly the self-collision its model defines."""
        used = {r.info.get("bit") for r in self.robots.values()}
        bit = next(b for b in self.ROBOT_BITS if b not in used)
        inst.info["bit"] = bit
        others = sum(self.ROBOT_BITS) & ~bit
        for g in spec.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            g.contype = int(g.contype) | bit
            g.conaffinity = int(g.conaffinity) | others

    def _pose_new(self, model, data, inst: Instance) -> None:
        """A freshly attached robot starts at its home configuration, at rest, with
        its position servos holding that configuration."""
        for name, q in inst.rm.home.items():
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, inst.prefix + name)
            if j < 0:
                continue
            data.qpos[model.jnt_qposadr[j]] = q
            data.qvel[model.jnt_dofadr[j]] = 0.0
        for a in range(model.nu):
            name = model.actuator(a).name
            if not name.startswith(inst.prefix):
                continue
            data.ctrl[a] = self.hold_value(model, data, a)
        if inst.rm.floating:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, inst.prefix + "root")
            if j >= 0:
                adr = model.jnt_qposadr[j]
                data.qpos[adr:adr + 7] = inst.rm.root_qpos(inst.xyz, inst.yaw)
                data.qvel[model.jnt_dofadr[j]:model.jnt_dofadr[j] + 6] = 0.0

    @staticmethod
    def hold_value(model, data, a: int) -> float:
        """The control that holds an actuator still: its joint position for a position
        servo, zero for a velocity or torque actuator."""
        if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT and \
                model.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE and \
                model.actuator_biasprm[a][1] != 0:
            j = model.actuator_trnid[a][0]
            q = float(data.qpos[model.jnt_qposadr[j]])
            lo, hi = model.actuator_ctrlrange[a]
            if model.actuator_ctrllimited[a]:
                q = min(max(q, lo), hi)
            return q
        return 0.0

    def remove(self, rid: str) -> Instance | None:
        with self.lock:
            inst = self.robots.pop(rid, None)
            if inst is None:
                return None
            if inst.body is not None:
                robot_model._delete(self.spec, inst.body)
                self._drop_prefixed(inst.prefix)
                if inst.staging is not None:
                    # its objects go with it; the scene objects cleared for them come back
                    inst.staging.undo(self.spec)
                back = inst.info.pop("displaced_scene_objects", False)
                if back:
                    self.scene_staging.add_objects(self.spec)   # the scene's own, again
                model, data = self.spec.recompile(self.model, self.data)
                if inst.staging is not None:
                    inst.staging.restore_state(model, data)
                if back:
                    self.scene_staging.place_objects(model, data)
                mujoco.mj_forward(model, data)
                self._swap(model, data)
            return inst

    def _drop_prefixed(self, prefix: str) -> None:
        """Remove the attached robot's leftover assets and frames (MuJoCo 3.5 through
        `spec.delete`, 3.3 through the element's own `delete`)."""
        for coll in ("meshes", "materials", "textures", "actuators", "sensors",
                     "equalities", "excludes", "tendons", "pairs"):
            for el in list(getattr(self.spec, coll, [])):
                if getattr(el, "name", "").startswith(prefix):
                    try:
                        if hasattr(self.spec, "delete"):
                            self.spec.delete(el)
                        elif hasattr(el, "delete"):
                            el.delete()
                    except Exception:
                        pass

    def _swap(self, model, data) -> None:
        self.model, self.data = model, data
        self.version += 1
        for fn in list(self.listeners):
            try:
                fn(self)
            except Exception as exc:  # a listener never breaks the world
                self.log(f"warning: recompile listener failed: {exc}")

    # ------------------------------------------------------------------ names

    def ids(self, kind, prefix: str, names):
        out = []
        for n in names:
            i = mujoco.mj_name2id(self.model, kind, prefix + n)
            if i < 0:
                raise KeyError(f"{prefix}{n}")
            out.append(i)
        return out

    def robot_elements(self, inst: Instance, kind, prefix_extra: str = ""):
        """[(local name, id)] of the robot's elements of a kind, in model order."""
        m = self.model
        count = {mujoco.mjtObj.mjOBJ_JOINT: m.njnt, mujoco.mjtObj.mjOBJ_ACTUATOR: m.nu,
                 mujoco.mjtObj.mjOBJ_CAMERA: m.ncam, mujoco.mjtObj.mjOBJ_SITE: m.nsite,
                 mujoco.mjtObj.mjOBJ_BODY: m.nbody, mujoco.mjtObj.mjOBJ_GEOM: m.ngeom}[kind]
        p = inst.prefix + prefix_extra
        out = []
        for i in range(count):
            name = mujoco.mj_id2name(m, kind, i) or ""
            if name.startswith(p):
                out.append((name[len(p):], i))
        return out

    def robot_body_ids(self, inst: Instance) -> set:
        m = self.model
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, inst.prefix + inst.rm.root)
        if root < 0:
            return set()
        ids = {root}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    ids.add(b)
                    break
                p = m.body_parentid[p]
        return ids

    # ------------------------------------------------------------------ physics loop

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="physics", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def call(self, fn, timeout: float = 10.0):
        """Run fn() on the physics thread between two steps, under the lock, and return
        its result. Samples taken this way are never delayed behind a burst of steps:
        a plain lock would let the stepping thread re-take it before a waiter wakes."""
        if threading.current_thread() is self._thread or self._thread is None \
                or not self._thread.is_alive():
            with self.lock:
                return fn()
        box = [threading.Event(), None, None]
        self._calls.put((fn, box))
        if not box[0].wait(timeout):
            raise TimeoutError("the physics thread did not answer")
        if box[2] is not None:
            raise box[2]
        return box[1]

    def set_ctrl(self, values: dict) -> None:
        """Actuator targets (full actuator name -> value) for the physics thread to apply
        before its next step. Never waits for the world lock: a wire streaming targets
        (100 Hz and more) through the plain lock queues behind the stepping thread, which
        re-takes it before a waiter wakes, and its commands then reach the physics late
        by up to the socket's backlog. A newer value for an actuator replaces one not yet
        applied."""
        if self._thread is None or not self._thread.is_alive():
            with self.lock:
                self._apply_ctrl(values)
            return
        with self._ctrl_lock:
            self._ctrl_pending.update(values)

    def _apply_ctrl(self, values: dict = None) -> None:
        """Under self.lock: write pending (or the given) actuator targets into data.ctrl,
        clamped to their ranges; names no longer in the model (a removed robot) are
        skipped."""
        if values is None:
            if not self._ctrl_pending:
                return
            with self._ctrl_lock:
                values, self._ctrl_pending = self._ctrl_pending, {}
        m = self.model
        for name, v in values.items():
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                continue
            v = float(v)
            if m.actuator_ctrllimited[a]:
                lo, hi = m.actuator_ctrlrange[a]
                v = min(max(v, lo), hi)
            self.data.ctrl[a] = v

    def snapshot(self) -> dict:
        """The configuration of the world now (small arrays, copied on the physics
        thread) for a `Mirror` to rebuild positions from, off the physics thread."""
        def take():
            d = self.data
            return {"model": self.model, "version": self.version, "qpos": d.qpos.copy(),
                    "mocap_pos": d.mocap_pos.copy(), "mocap_quat": d.mocap_quat.copy(),
                    "time": float(d.time), "stamp": time.time()}
        return self.call(take)

    def _serve_calls(self, block: float = 0.0) -> None:
        try:
            item = self._calls.get(timeout=block) if block > 0 else self._calls.get_nowait()
        except queue.Empty:
            return
        while item is not None:
            fn, box = item
            try:
                with self.lock:
                    box[1] = fn()
            except Exception as exc:
                box[2] = exc
            box[0].set()
            try:
                item = self._calls.get_nowait()
            except queue.Empty:
                item = None

    def _run(self) -> None:
        wall0 = time.monotonic()
        with self.lock:
            sim0 = self.data.time
        win_wall, win_sim = wall0, sim0
        while not self._stop.is_set():
            now = time.monotonic()
            target = sim0 + (now - wall0)
            steps = 0
            while True:
                self._serve_calls()
                with self.lock:
                    d = self.data
                    if d.time >= target:
                        break
                    self._apply_ctrl()
                    # a few steps per call: MuJoCo releases the GIL while it steps, and
                    # fewer round trips through Python keep other threads from starving
                    # the physics (calls are still served every few milliseconds)
                    n = int(min(STEP_BATCH, max(1, math.ceil((target - d.time)
                                                                / self.model.opt.timestep))))
                    mujoco.mj_step(self.model, d, nstep=n)
                    if not np.isfinite(d.qacc).all():
                        self.log("warning: physics diverged; clearing velocities")
                        d.qvel[:] = 0
                        d.qacc_warmstart[:] = 0
                steps += 1
                if steps >= 200:
                    break
                if time.monotonic() - now > 0.05:
                    break
            with self.lock:
                simt = self.data.time
            now = time.monotonic()
            if target - simt > 0.25:
                # Cannot keep up: let simulated time fall behind rather than bursting to
                # catch up, and let the real-time factor say so.
                wall0, sim0 = now, simt
            if now - win_wall >= RTF_WINDOW_S:
                self.rtf = (simt - win_sim) / (now - win_wall)
                if self.rtf < RTF_WARN:
                    self.log(f"warning: real-time factor {self.rtf:.2f} over the last "
                             f"{now - win_wall:.0f} s is below {RTF_WARN:.2f}; rates and "
                             "physical bounds are not met for this interval")
                win_wall, win_sim = now, simt
            with self.lock:
                ahead = self.data.time - (sim0 + (time.monotonic() - wall0))
            if ahead > 0:
                self._serve_calls(block=min(ahead, 0.005))
