#!/usr/bin/env python
"""The simulation process: one engine's scene, its physics, its rendering, its robots.

    python   simulation.py --engine molmospaces [--scene ithor:1] [--sim-port 9080]
    mjpython simulation.py --engine robocasa --mujoco        # MuJoCo window (macOS)

Started by `simulator/<engine>/run.sh start` (and `simulator/kitchen.sh start`). It loads
the scene alone, with no robot, and accepts on the local control port `--sim-port`:
spawn and removal requests from `spawn.sh`, the wire containers' command and sensor
traffic, and simulator-private requests (an offscreen render from a viewpoint, a robot's
joint and pose readings). It knows nothing of ROS. See simulator/README.md for the
protocol.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import queue
import secrets
import signal
import socket
import sys
import threading
import time
import traceback

import mujoco
import numpy as np

import placement as placement_mod
import protocol
import registry
import robot_model
import scenes
import sensors
import worktop_objects
from world import Instance, World

READY_PREFIX = "simulation ready"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- connections


class Conn:
    """One control-port connection, with a writer thread so a slow reader never blocks
    the physics or the sampler. Samples of a stream are dropped, oldest first, when the
    reader falls behind (never queued without bound)."""

    def __init__(self, server, sock, addr):
        self.server = server
        self.sock = sock
        self.addr = addr
        self.id = next(server.conn_ids)
        self.role = "client"
        self.instance: Instance | None = None      # the robot this conn leases (spawn)
        self.wire_of: Instance | None = None       # the robot this wire serves
        self.component = None
        self.outq: queue.Queue = queue.Queue()
        self.pending_samples: dict[int, int] = {}
        self.plock = threading.Lock()
        self.subs: dict[int, dict] = {}
        self.closed = threading.Event()
        self.writer = threading.Thread(target=self._write, daemon=True, name=f"conn{self.id}-w")
        self.writer.start()

    def send(self, header, payload=None, sub=None):
        if self.closed.is_set():
            return
        if sub is not None:
            with self.plock:
                n = self.pending_samples.get(sub, 0)
                if n >= 2:
                    return  # the reader is behind: drop this sample
                self.pending_samples[sub] = n + 1
        self.outq.put((header, payload, sub))

    def _write(self):
        while True:
            item = self.outq.get()
            if item is None:
                return
            header, payload, sub = item
            try:
                protocol.send(self.sock, header, payload)
            except OSError:
                self.close()
                return
            finally:
                if sub is not None:
                    with self.plock:
                        self.pending_samples[sub] = max(0, self.pending_samples.get(sub, 1) - 1)

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass
        self.outq.put(None)


# ---------------------------------------------------------------- the server


class Simulation:
    def __init__(self, args):
        self.args = args
        self.engine = args.engine
        self.conn_ids = itertools.count(1)
        self.sub_ids = itertools.count(1)
        self.conns: dict[int, Conn] = {}
        self.clock = threading.Lock()          # admission and connection bookkeeping
        self.starting: str | None = None       # the id whose startup is in progress
        self.shutdown_event = threading.Event()
        self.exit_code = 0
        self.world: World | None = None
        self.surfaces = None
        self.renders = None
        self.listener = None

    # -------------------------------------------------------------- startup

    def bind(self):
        port = self.args.sim_port
        taken = None
        # A port someone listens on (any interface) refuses the start. SO_REUSEADDR only
        # lets a restart bind over the TIME_WAIT leftovers of a simulation just ended.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(0.5)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            taken = "something accepts connections there"
        probe.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if taken is None:
            try:
                s.bind(("127.0.0.1", port))
            except OSError as exc:
                taken = exc.strerror
        if taken is not None:
            raise SystemExit(f"error: --sim-port {port} is already in use ({taken}); is a "
                             "simulation already running there?")
        s.listen(64)
        self.listener = s

    def load(self):
        t0 = time.time()
        try:
            scene = scenes.load(self.engine, self.args.scene)
        except scenes.SceneError as exc:
            raise SystemExit(f"error: {exc}")
        self.world = World(scene, log=log)
        m = self.world.model
        log(f"{scene.name} on {self.engine}: {m.nbody} bodies, {m.ngeom} geoms, "
            f"{m.nmesh} meshes, {m.nlight} lights ({time.time() - t0:.1f} s)")
        t1 = time.time()
        with self.world.lock:
            # the survey reads the bare scene as loaded, whatever the physics does later
            bare = copy.copy(self.world.data)
            facts = scene.survey_facts(self.world.model, bare)
            self.surfaces = placement_mod.SceneSurfaces(self.world.model, bare,
                                                         self.world.scene.geomgroup, facts=facts)
        wt = self.surfaces.worktop
        log(f"floor at z {self.surfaces.floor_z:.3f}; worktop: "
            + (json.dumps(wt.describe()) if wt else "none") + f" ({time.time() - t1:.1f} s)")
        self._stage_worktop_objects(wt)
        self.renders = sensors.RenderPool(self.world, threads=int(os.environ.get(
            "RSIM_RENDER_THREADS", "3")))
        self.world.start()
        threading.Thread(target=self._accept, daemon=True, name="accept").start()
        threading.Thread(target=self._sampler, daemon=True, name="sampler").start()

    def _stage_worktop_objects(self, wt) -> None:
        """The six worktop objects are part of the scene (spec §2.2), whatever robots come:
        staged around the spot the SO-101 -- the reference's own arm -- is placed at on
        the worktop, with the loose scene objects in their working area cleared. A scene
        with no worktop has none."""
        if wt is None:
            return
        t0 = time.time()
        try:
            rm = robot_model.load(registry.get("so101"))
            pl = placement_mod.place(self.world, self.surfaces, rm, "worktop", False, "so101/")
            self.world.stage_scene(pl.staging)
        except (placement_mod.Refused, robot_model.ModelError, registry.RegistryError,
                worktop_objects.AssetsMissing) as exc:
            log(f"no worktop objects staged: {exc}")
            return
        log(f"worktop objects staged: {', '.join(worktop_objects.OBJECTS)}; scene objects "
            f"cleared from their area: {', '.join(pl.staging.cleared) or 'none'} "
            f"({time.time() - t0:.1f} s)")

    def ready_line(self) -> str:
        return (f"{READY_PREFIX}: engine {self.engine}, scene {self.world.scene.name}, "
                f"sim-port {self.args.sim_port}, "
                f"{'window' if self.args.mujoco else 'headless'}")

    # -------------------------------------------------------------- accept/read

    def _accept(self):
        while not self.shutdown_event.is_set():
            try:
                sock, addr = self.listener.accept()
            except OSError:
                return
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn = Conn(self, sock, addr)
            with self.clock:
                self.conns[conn.id] = conn
            threading.Thread(target=self._serve, args=(conn,), daemon=True,
                             name=f"conn{conn.id}").start()

    def _serve(self, conn: Conn):
        try:
            while not conn.closed.is_set():
                header, payload = protocol.recv(conn.sock)
                op = header.get("op")
                rid = header.get("id")
                try:
                    handler = getattr(self, f"op_{op}", None)
                    if handler is None:
                        raise ValueError(f"unknown op {op!r}")
                    result = handler(conn, header, payload)
                    if rid is not None:
                        if isinstance(result, tuple):
                            conn.send(dict(result[0], id=rid, ok=True), result[1])
                        else:
                            conn.send(dict(result or {}, id=rid, ok=True))
                except Exception as exc:  # every failure is answered, never fatal
                    if not isinstance(exc, (ValueError, KeyError, placement_mod.Refused,
                                            robot_model.ModelError, registry.RegistryError,
                                            worktop_objects.AssetsMissing)):
                        traceback.print_exc()
                    if rid is not None:
                        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
                        conn.send({"id": rid, "ok": False, "error": str(msg)})
        except (ConnectionError, OSError, protocol.ProtocolError, ValueError):
            pass
        finally:
            self._dropped(conn)

    def _dropped(self, conn: Conn):
        conn.close()
        with self.clock:
            self.conns.pop(conn.id, None)
        for sub in list(conn.subs.values()):
            if "pool" in sub:
                sub["pool"].close()
        inst = conn.instance
        if inst is not None and not self.shutdown_event.is_set():
            # The spawn that owned this robot is gone (ended, failed or killed):
            # remove its robot and release its id, ports and placement.
            self._remove(inst, reason="its spawn ended" if inst.state == "running"
                         else "its startup ended")
        if conn.wire_of is not None:
            with self.clock:
                if conn in conn.wire_of.wires:
                    conn.wire_of.wires.remove(conn)

    # -------------------------------------------------------------- robot lifecycle

    def _instance(self, rid: str) -> Instance:
        inst = self.world.robots.get(rid)
        if inst is None:
            raise KeyError(f"no robot {rid!r} in this simulation "
                           f"(spawned: {', '.join(self.world.robots) or 'none'})")
        return inst

    def _remove(self, inst: Instance, reason: str):
        with self.clock:
            if self.starting == inst.id:
                self.starting = None
            wires = list(inst.wires)
            inst.wires.clear()
        removed = self.world.remove(inst.id)
        for w in wires:
            w.send({"event": "removed", "robot": inst.id, "reason": reason})
            threading.Timer(0.2, w.close).start()
        if removed is not None:
            log(f"robot {inst.id} removed ({reason})")

    def op_hello(self, conn, h, p):
        conn.role = h.get("role", "client")
        wt = self.surfaces.worktop
        with self.world.lock:
            t = float(self.world.data.time)
        return {"engine": self.engine, "scene": self.world.scene.name, "sim_time": t,
                "sim_port": self.args.sim_port, "window": bool(self.args.mujoco),
                "floor_z": self.surfaces.floor_z,
                "worktop": wt.describe() if wt else None,
                "robots": self._robot_rows(), "starting": self.starting,
                "rtf": round(self.world.rtf, 3)}

    def _robot_rows(self):
        rows = []
        for inst in list(self.world.robots.values()):
            rows.append({"id": inst.id, "state": inst.state, "placement": inst.placement,
                         "xyz": [round(float(v), 4) for v in inst.xyz],
                         "yaw": round(float(inst.yaw), 4), "ports": list(inst.ports),
                         "prefix": inst.prefix,
                         "components": [{"role": c.role, "robot": c.robot.id,
                                         "prefix": c.prefix} for c in inst.rm.components],
                         **self._staging_of(inst)})
        return rows

    def _staging_of(self, inst) -> dict:
        """The worktop objects an arm stands among, as staged (`staged`: {name: {body,
        pos, quat}}), and the scene objects cleared for them (`cleared`: body names)."""
        st = inst.staging
        if st is None and inst.info.get("at_scene_objects"):
            st = self.world.scene_staging
        if st is None:
            return {"staged": {}, "cleared": []}
        return {"staged": {n: {"body": worktop_objects.BODIES[n], "pos": p, "quat": q}
                           for n, (p, q) in st.poses().items()},
                "cleared": list(st.cleared)}

    def op_robots(self, conn, h, p):
        return {"robots": self._robot_rows(), "starting": self.starting}

    def _admission(self, rid, where, ports):
        """Refusals that need no model: duplicate id, busy startup, reserved ports and an
        occupied worktop. Called with self.clock held."""
        if rid in self.world.robots:
            raise ValueError(f"refused: robot {rid!r} is already spawned in this "
                             f"simulation ({self.world.robots[rid].state}); end that "
                             "spawn first")
        if self.starting is not None:
            raise ValueError(f"refused: another spawn ({self.starting}) is starting "
                             "up; try again once it is ready")
        for other in self.world.robots.values():
            clash = set(ports) & set(other.ports)
            if clash:
                raise ValueError(f"refused: port {sorted(clash)[0]} is reserved by "
                                 f"the running {other.id}")
        if where == "worktop":
            if self.surfaces.worktop is None:
                raise ValueError("placement refused: this scene has no worktop ("
                                 + placement_mod.NO_WORKTOP + ")")
            holder = next((o.id for o in self.world.robots.values()
                           if o.placement == "worktop"), None)
            if holder:
                raise ValueError(f"placement refused: the worktop already holds {holder}")

    def op_precheck(self, conn, h, p):
        """The admission checks of `spawn`, reserving nothing (spawn.sh asks before it
        spends time building wire images)."""
        try:
            registry.get(h.get("robot", ""))
        except registry.RegistryError as exc:
            raise ValueError(str(exc))
        with self.clock:
            self._admission(h.get("robot", ""), h.get("placement", "worktop"),
                            tuple(int(x) for x in h.get("ports", [])))
        return {}

    def op_spawn(self, conn, h, p):
        rid = h.get("robot", "")
        where = h.get("placement", "worktop")
        ports = tuple(int(x) for x in h.get("ports", []))
        if conn.instance is not None:
            raise ValueError("this connection already holds a spawn")
        try:
            robot = registry.get(rid)
        except registry.RegistryError as exc:
            raise ValueError(str(exc))
        with self.clock:
            self._admission(rid, where, ports)
            self.starting = rid
        try:
            rm = robot_model.load(robot)
            prefix = f"{rid}/"
            t0 = time.time()
            pl = placement_mod.place(self.world, self.surfaces, rm, where, robot.mobile, prefix)
            inst = Instance(id=rid, rm=rm, prefix=prefix, placement=where, xyz=pl.xyz,
                            yaw=pl.yaw, support_geoms=pl.support_geoms, ports=ports,
                            token=secrets.token_hex(8), lease=conn)
            inst.info["at_scene_objects"] = pl.at_scene_objects
            with self.clock:
                if conn.closed.is_set():
                    raise ValueError("spawn connection closed during startup")
                self.world.add(inst, staging=pl.staging)
                conn.instance = inst
            log(f"robot {rid} added: {pl.report[0]} ({time.time() - t0:.1f} s)")
        except Exception:
            with self.clock:
                if self.starting == rid:
                    self.starting = None
            raise
        return {"robot": rid, "token": inst.token, "prefix": prefix,
                "xyz": [float(v) for v in inst.xyz], "yaw": float(inst.yaw),
                "surface_z": pl.surface_z, "placement": where,
                "components": [{"role": c.role, "robot": c.robot.id, "prefix": c.prefix}
                               for c in rm.components],
                **self._staging_of(inst)}

    def op_commit(self, conn, h, p):
        inst = conn.instance
        if inst is None:
            raise ValueError("no spawn on this connection")
        with self.clock:
            inst.state = "running"
            if self.starting == inst.id:
                self.starting = None
        log(f"robot {inst.id} running on port(s) {', '.join(map(str, inst.ports))}")
        return {}

    def op_remove(self, conn, h, p):
        inst = conn.instance
        if inst is None:
            raise ValueError("no spawn on this connection")
        conn.instance = None
        self._remove(inst, reason=h.get("reason", "its spawn ended"))
        return {}

    def op_wire(self, conn, h, p):
        """A wire container attaches to the robot whose spawn token it carries."""
        tok = h.get("token")
        for inst in list(self.world.robots.values()):
            if inst.token == tok:
                role = h.get("role", "main")
                comp = next((c for c in inst.rm.components if c.role == role), None)
                if comp is None:
                    raise ValueError(f"{inst.id} has no {role!r} component")
                conn.role, conn.wire_of, conn.component = "wire", inst, comp
                with self.clock:
                    inst.wires.append(conn)
                return {"robot": inst.id, "component": comp.robot.id,
                        "describe": self._describe(inst, comp)}
        raise ValueError("unknown spawn token (the robot is gone)")

    # -------------------------------------------------------------- model facts

    def _describe(self, inst, comp) -> dict:
        w = self.world
        m = w.model
        p = comp.prefix
        joints, acts, cams, sites = [], [], [], []
        with w.lock:
            for name, j in w.robot_elements(inst, mujoco.mjtObj.mjOBJ_JOINT, p):
                if comp.role == "base" and name.startswith(robot_model.ARM_PREFIX):
                    continue
                joints.append({"name": name, "type": int(m.jnt_type[j]),
                               "range": [float(x) for x in m.jnt_range[j]],
                               "limited": bool(m.jnt_limited[j])})
            for name, a in w.robot_elements(inst, mujoco.mjtObj.mjOBJ_ACTUATOR, p):
                if comp.role == "base" and name.startswith(robot_model.ARM_PREFIX):
                    continue
                kind = "position" if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE \
                    and m.actuator_biasprm[a][1] != 0 else \
                    ("velocity" if m.actuator_biastype[a] == mujoco.mjtBias.mjBIAS_AFFINE
                     and m.actuator_biasprm[a][2] != 0 else "motor")
                acts.append({"name": name, "kind": kind,
                             "ctrlrange": [float(x) for x in m.actuator_ctrlrange[a]],
                             "limited": bool(m.actuator_ctrllimited[a])})
            for name, c in w.robot_elements(inst, mujoco.mjtObj.mjOBJ_CAMERA, p):
                if comp.role == "base" and name.startswith(robot_model.ARM_PREFIX):
                    continue
                res = [int(x) for x in m.cam_resolution[c]] if hasattr(m, "cam_resolution") else [0, 0]
                cams.append({"name": name, "fovy": float(m.cam_fovy[c]), "resolution": res})
            for name, s in w.robot_elements(inst, mujoco.mjtObj.mjOBJ_SITE, p):
                if comp.role == "base" and name.startswith(robot_model.ARM_PREFIX):
                    continue
                sites.append(name)
        return {"joints": joints, "actuators": acts, "cameras": cams, "sites": sites,
                "root": inst.rm.root if comp.prefix == "" else None,
                "floating": inst.rm.floating, "timestep": float(m.opt.timestep)}

    def op_describe(self, conn, h, p):
        inst = self._instance(h["robot"])
        role = h.get("component", inst.rm.components[0].role)
        comp = next(c for c in inst.rm.components if c.role == role)
        return self._describe(inst, comp)

    # -------------------------------------------------------------- readings, render

    def op_readings(self, conn, h, p):
        """A spawned robot's joint and pose readings (simulator-private)."""
        inst = self._instance(h["robot"])
        w = self.world
        with w.lock:
            joints = sensors.joint_readings(w, inst)
            base = sensors.base_reading(w, inst)
            bodies = {}
            for name, b in w.robot_elements(inst, mujoco.mjtObj.mjOBJ_BODY):
                bodies[name] = sensors.body_pose(w, b)
            sites = sensors.site_readings(w, inst, [n for n, _ in w.robot_elements(
                inst, mujoco.mjtObj.mjOBJ_SITE)])
            ctrl = {name: float(w.data.ctrl[a]) for name, a in
                    w.robot_elements(inst, mujoco.mjtObj.mjOBJ_ACTUATOR)}
            t = float(w.data.time)
        return {"robot": inst.id, "stamp": time.time(), "sim_time": t,
                "joints": {k: {"position": v[0], "velocity": v[1], "effort": v[2]}
                           for k, v in joints.items()},
                "base": base, "bodies": bodies, "sites": sites, "ctrl": ctrl}

    def op_bodies(self, conn, h, p):
        """World poses of named bodies (scene objects included)."""
        w = self.world
        out = {}
        with w.lock:
            for name in h.get("names", []):
                b = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if b >= 0:
                    out[name] = sensors.body_pose(w, b)
            t = float(w.data.time)
            state = None
            if h.get("state"):
                state = {"qpos_sha": hash(w.data.qpos.tobytes()), "nq": int(w.model.nq)}
        return {"bodies": out, "sim_time": t, "state": state}

    def op_scene(self, conn, h, p):
        """Scene facts: model counts, the scene-only portion, bodies with free joints."""
        w = self.world
        with w.lock:
            m = w.model
            free = []
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    b = m.jnt_bodyid[j]
                    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"
                    if not any(name.startswith(r.prefix) for r in w.robots.values()):
                        free.append(name)
            staging = {r.id: self._staging_of(r) for r in w.robots.values()
                       if r.staging is not None}
            if w.scene_staging is not None:
                staging["scene"] = {"staged": {n: {"body": worktop_objects.BODIES[n], "pos": p,
                                                   "quat": q}
                                               for n, (p, q) in w.scene_staging.poses().items()},
                                    "cleared": list(w.scene_staging.cleared)}
            return {"nbody": m.nbody, "ngeom": m.ngeom, "nmesh": m.nmesh, "nq": m.nq,
                    "nu": m.nu, "ncam": m.ncam, "nlight": m.nlight,
                    "scene_nbody": w.scene_nbody, "scene_ngeom": w.scene_ngeom,
                    "free_bodies": free, "staging": staging,
                    "floor_z": self.surfaces.floor_z,
                    "worktop": self.surfaces.worktop.describe() if self.surfaces.worktop else None,
                    "sim_time": float(w.data.time), "rtf": round(w.rtf, 3),
                    "timestep": float(m.opt.timestep)}

    def _frame_robot(self, rid: str, elevation: float = -30.0) -> dict:
        """A free-camera viewpoint that shows the whole robot -- and the objects staged
        with it -- with an unobstructed line of sight: looking at their bounding-box centre
        from in front of the robot, trying azimuths and distances until the ray from the
        robot to the eye is clear."""
        inst = self._instance(rid)
        w = self.world

        def pick():
            m, d = w.model, w.data
            own = w.robot_body_ids(inst)
            pts = [d.xpos[b] for b in own]
            if inst.staging is not None or inst.info.get("at_scene_objects"):
                for body in worktop_objects.BODIES.values():
                    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
                    if b >= 0:
                        pts.append(d.xpos[b])
                        own = own | {b}
            lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
            centre = (lo + hi) / 2
            size = max(float(np.linalg.norm(hi - lo)), 0.25)
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, inst.prefix + inst.rm.root)
            R = d.xmat[root].reshape(3, 3)
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            gid = np.zeros(1, np.int32)
            best = None
            for dist in (max(0.9, 2.2 * size), max(0.7, 1.6 * size), 0.6):
                for daz in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                    az = yaw + daz
                    el = math.radians(elevation)
                    back = np.array([math.cos(math.radians(az)) * math.cos(el),
                                     math.sin(math.radians(az)) * math.cos(el),
                                     -math.sin(el)])
                    # clear from the robot's centre out to the eye (and a little beyond),
                    # looking past the robot's own geometry
                    origin, travelled, exclude = centre.copy(), 0.0, root
                    blocked = False
                    for _ in range(12):
                        hit = mujoco.mj_ray(m, d, origin, back, None, 1, exclude, gid)
                        if hit < 0 or travelled + hit >= dist + 0.15:
                            break
                        if m.geom_bodyid[gid[0]] in own:
                            exclude = int(m.geom_bodyid[gid[0]])
                            origin = origin + back * (hit + 1e-4)
                            travelled += hit + 1e-4
                            continue
                        blocked = True
                        break
                    if not blocked:
                        return {"lookat": [float(x) for x in centre], "distance": dist,
                                "azimuth": az + 180.0, "elevation": elevation}
                    free = travelled + hit
                    if best is None or free > best[0]:
                        best = (free, {"lookat": [float(x) for x in centre],
                                       "distance": max(0.3, free - 0.1),
                                       "azimuth": az + 180.0, "elevation": elevation})
            # nowhere fully clear: the direction with the longest clear line, inside it
            return best[1]

        return w.call(pick)

    def op_render(self, conn, h, p):
        """An offscreen render of the scene from a viewpoint (simulator-private)."""
        width, height = int(h.get("width", 1280)), int(h.get("height", 720))
        fmt = h.get("format", "png")
        view = h.get("view") or {k: h[k] for k in ("camera", "lookat", "distance", "azimuth",
                                                   "elevation", "pos", "target", "fovy")
                                 if k in h}
        if view.get("frame_robot") or ("robot" in h and not view):
            view = self._frame_robot(view.get("frame_robot") or h["robot"],
                                     float(view.get("elevation", -30.0)))
        if "robot_camera" in h:
            view = {"camera": f"{h['robot']}/{h['robot_camera']}"}
        if "camera" in view and "/" not in view["camera"] and h.get("robot"):
            view = {"camera": f"{h['robot']}/{view['camera']}"}
        job = sensors.RenderJob(view, width, height, rgb=fmt in ("png", "rgb"),
                                depth=fmt == "depth")
        res = self.renders.render_sync(job)
        meta = {"width": width, "height": height, "format": fmt, "stamp": res["stamp"],
                "sim_time": res["sim_time"]}
        if fmt == "png":
            return meta, sensors.png_bytes(res["rgb"])
        if fmt == "rgb":
            return meta, res["rgb"].tobytes()
        return meta, res["depth"].tobytes()

    # -------------------------------------------------------------- wire traffic

    def _wire(self, conn) -> Instance:
        if conn.wire_of is None:
            raise ValueError("not a wire connection")
        return conn.wire_of

    def op_ctrl(self, conn, h, p):
        """Set actuator targets by actuator (joint) name, in the actuator's units. Unknown
        names are refused at once; the targets reach the physics before its next step
        (World.set_ctrl), without this connection waiting for the world lock."""
        inst = self._wire(conn)
        pre = inst.prefix + conn.component.prefix
        m = self.world.model
        values = {}
        for name, v in (h.get("values") or {}).items():
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, pre + name) < 0:
                raise KeyError(f"no actuator {name!r}")
            values[pre + name] = float(v)
        self.world.set_ctrl(values)
        return {}

    def op_subscribe(self, conn, h, p):
        """A periodic stream to this connection: `state`, `camera` or `lidar`."""
        inst = self._wire(conn) if conn.wire_of is not None else self._instance(h["robot"])
        kind = h["stream"]
        rate = float(h["rate"])
        if rate <= 0:
            raise ValueError("rate must be positive")
        sub = {"kind": kind, "period": 1.0 / rate, "next": time.monotonic(), "inst": inst,
               "conn": conn, "params": h, "busy": 0, "last": 0.0,
               "prefix": conn.component.prefix if conn.component else ""}
        if kind == "camera":
            cam = f"{inst.prefix}{sub['prefix']}{h['camera']}"
            if mujoco.mj_name2id(self.world.model, mujoco.mjtObj.mjOBJ_CAMERA, cam) < 0:
                raise KeyError(f"no camera {h['camera']!r} on {inst.id}")
            # Each camera stream renders on its own pair of threads with a buffer of its
            # own size, so one stream never waits behind another or a scene render.
            sub["pool"] = sensors.RenderPool(self.world, threads=2,
                                             max_size=(int(h["width"]), int(h["height"])))
        elif kind not in ("state", "lidar"):
            raise ValueError(f"unknown stream {kind!r}")
        sid = next(self.sub_ids)
        with self.clock:
            conn.subs[sid] = sub
        return {"sub": sid}

    def _sampler(self):
        """Fires every due subscription at its rate, by elapsed wall time."""
        while not self.shutdown_event.is_set():
            now = time.monotonic()
            nxt = now + 0.01
            with self.clock:
                subs = [(sid, s) for c in self.conns.values() for sid, s in c.subs.items()
                        if not c.closed.is_set()]
            for sid, s in subs:
                if s["next"] <= now:
                    # a camera may have two renders in flight; other streams are sampled
                    # synchronously
                    if s["busy"] < (2 if s["kind"] == "camera" else 1):
                        try:
                            self._fire(sid, s)
                        except Exception as exc:
                            log(f"warning: stream {s['kind']} failed: {exc}")
                    s["next"] += s["period"]
                    if s["next"] < now - s["period"]:
                        s["next"] = now + s["period"]
                nxt = min(nxt, s["next"])
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(min(delay, 0.01))

    def _fire(self, sid, s):
        conn, inst, prm, pre = s["conn"], s["inst"], s["params"], s["prefix"]
        w = self.world
        if inst.id not in w.robots:
            return
        kind = s["kind"]
        if kind == "state":
            def sample():
                out = {"event": "sample", "sub": sid, "stamp": time.time(),
                       "sim_time": float(w.data.time)}
                if prm.get("joints", True):
                    j = sensors.joint_readings(w, inst, pre)
                    if pre == "":
                        j = {k: v for k, v in j.items()
                             if not k.startswith(robot_model.ARM_PREFIX)}
                    out["joints"] = j
                if prm.get("base", False) and pre == "":
                    out["base"] = sensors.base_reading(w, inst)
                strip = lambda d: {k[len(pre):]: v for k, v in d.items()}
                if prm.get("sites"):
                    out["sites"] = strip(sensors.site_readings(
                        w, inst, [pre + n for n in prm["sites"]]))
                if prm.get("imu_sites"):
                    out["imu"] = strip(sensors.site_readings(
                        w, inst, [pre + n for n in prm["imu_sites"]], imu=True))
                return out

            conn.send(w.call(sample), sub=sid)
        elif kind == "lidar":
            # The world is sampled now (a snapshot on the physics thread); the rays are
            # cast on this stream's own mirror of it, on its own thread.
            snap = w.snapshot()
            s["busy"] += 1

            def scan(snap=snap, s=s, sid=sid):
                try:
                    from world import Mirror

                    mirror = s.setdefault("mirror", Mirror())
                    m, d = mirror.update(snap)
                    if s.get("own_version") != snap["version"]:
                        s["own"] = sensors.subtree(m, inst.prefix + inst.rm.root)
                        s["own_version"] = snap["version"]
                    r = sensors.lidar_scan(m, d, inst, pre + prm["site"],
                                           float(prm["angle_min"]), float(prm["angle_max"]),
                                           int(prm["samples"]), float(prm["range_min"]),
                                           float(prm["range_max"]), own=s["own"])
                    conn.send({"event": "sample", "sub": sid, "stamp": snap["stamp"],
                               "sim_time": snap["time"], "n": int(r.size)}, r.tobytes(),
                              sub=sid)
                except Exception as exc:
                    log(f"warning: lidar scan failed: {exc}")
                finally:
                    s["busy"] -= 1

            threading.Thread(target=scan, daemon=True, name="lidar").start()
        elif kind == "camera":
            s["busy"] += 1
            width, height = int(prm["width"]), int(prm["height"])
            want_rgb, want_depth = bool(prm.get("rgb", True)), bool(prm.get("depth", False))

            def done(res, sid=sid, s=s):
                s["busy"] -= 1
                if isinstance(res, Exception):
                    log(f"warning: camera render failed: {res}")
                    return
                if res["stamp"] <= s["last"]:
                    return  # overtaken by a newer frame: never send samples out of order
                s["last"] = res["stamp"]
                parts = []
                if want_rgb:
                    parts.append(res["rgb"].tobytes())
                if want_depth:
                    parts.append(res["depth"].tobytes())
                conn.send({"event": "sample", "sub": sid, "stamp": res["stamp"],
                           "sim_time": res["sim_time"], "width": width, "height": height,
                           "rgb": want_rgb, "depth": want_depth}, b"".join(parts), sub=sid)

            s["pool"].submit(sensors.RenderJob(
                {"camera": f"{inst.prefix}{pre}{prm['camera']}"}, width, height,
                rgb=want_rgb, depth=want_depth, callback=done))

    # -------------------------------------------------------------- shutdown

    def shutdown(self, reason: str):
        if self.shutdown_event.is_set():
            return
        self.shutdown_event.set()
        log(f"simulation ending: {reason}")
        with self.clock:
            conns = list(self.conns.values())
        for c in conns:
            c.send({"event": "shutdown", "reason": reason})
        time.sleep(0.3)
        for c in conns:
            c.close()
        try:
            self.listener.close()
        except OSError:
            pass
        if self.world is not None:
            self.world.stop()


# ---------------------------------------------------------------- viewer


def run_viewer(sim: Simulation):
    """The MuJoCo window, on the main thread (mjpython on macOS). Closing it ends the
    simulation; a robot added or removed reloads the window's model in place."""
    from mujoco import viewer as mj_viewer

    w = sim.world
    with w.lock:
        handle = mj_viewer.launch_passive(w.model, w.data)
        version = w.version
    gg = w.scene.geomgroup
    with handle.lock():
        for i in range(6):
            handle.opt.geomgroup[i] = int(gg[i]) if i < len(gg) else 0
    while handle.is_running() and not sim.shutdown_event.is_set():
        with w.lock:
            if w.version != version:
                sim_obj = handle._get_sim()
                if sim_obj is not None:
                    sim_obj.load(w.model, w.data, "")
                    handle._sim = (lambda s=sim_obj: s)
                version = w.version
            handle.sync()
        time.sleep(1 / 60)
    handle.close()
    if not sim.shutdown_event.is_set():
        sim.shutdown("the MuJoCo window was closed")


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="simulation.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", required=True, choices=sorted(scenes.ENGINES))
    ap.add_argument("--scene", default=None)
    ap.add_argument("--sim-port", type=int, default=protocol.DEFAULT_SIM_PORT, dest="sim_port")
    ap.add_argument("--mujoco", action="store_true")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # Many threads (physics, sampler, renderers, connection writers) share the GIL; a
    # short switch interval keeps the periodic streams on time.
    sys.setswitchinterval(0.0005)
    try:
        scenes.parse(args.engine, args.scene)
    except scenes.SceneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sim = Simulation(args)
    sim.bind()

    def on_signal(signum, frame):
        sim.shutdown(f"signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, on_signal)
    try:
        sim.load()
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    print(sim.ready_line(), flush=True)
    if args.mujoco:
        run_viewer(sim)
    else:
        while not sim.shutdown_event.is_set():
            sim.shutdown_event.wait(0.5)
    # give spawns the shutdown event before the process goes
    time.sleep(0.2)
    return sim.exit_code


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
