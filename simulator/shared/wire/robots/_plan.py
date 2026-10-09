"""Wire-plan helpers: which processes a robot's ROS graph runs (stock packages, where the
recorded boot runs them and they need no hardware; simulated drivers for the rest)."""

from __future__ import annotations

from pathlib import Path

WIRE = Path(__file__).resolve().parents[1]


class Plan:
    def __init__(self):
        self.procs = []     # (label, command)
        self.params = {}    # ROS 1 parameter name -> value, overriding / supplying ros.yml values
        self.prepare = []   # shell commands run (in order, to completion) before the graph
        self.overlays = []  # setup.bash files sourced for every process

    def add(self, label, cmd):
        self.procs.append((label, [str(c) for c in cmd]))


def ros1_emulated(plan: Plan, node: str):
    plan.add(node, ["python3", "-u", str(WIRE / "ros1_node.py"), node])


def ros1_name(node: str) -> list:
    """rosrun remapping arguments that give a stock node its recorded name."""
    ns, _, base = node.rpartition("/")
    args = [f"__name:={base}"]
    if ns:
        args.append(f"__ns:={ns}")
    return args


def ros1_static_tf(plan: Plan, iface: dict, package: str = "tf"):
    """Every non-optional static transform whose publisher is a `static_transform_publisher`
    node of the boot, run as that stock node with the recorded pose and period."""
    nodes = {n["name"]: n for n in iface.get("nodes", []) if not n.get("optional")}
    for row in iface.get("tf", []):
        if row.get("optional") or not row.get("static"):
            continue
        node = nodes.get(row.get("publisher"))
        if node is None or node.get("executable") != "static_transform_publisher":
            continue
        if node["package"] == "tf":
            # periodic on /tf: served on a fixed schedule so the recorded rate holds
            # (robots/_ros1_drivers.StaticTransformPublisher)
            ros1_emulated(plan, node["name"])
            continue
        x, y, z = row.get("xyz", [0, 0, 0])
        r, p, yw = row.get("rpy", [0, 0, 0])
        cmd = ["rosrun", node["package"], "static_transform_publisher",
               x, y, z, yw, p, r, row["parent"], row["child"]]
        plan.add(node["name"], cmd + ros1_name(node["name"]))


def _lx_plus_ly(k: dict) -> float:
    """lx + ly of the interface file's `kinematics` row: its own `lx_plus_ly` where it
    records one (a firmware constant), else the sum."""
    return float(k.get("lx_plus_ly") or float(k["lx"]) + float(k["ly"]))


def mecanum_rim(k: dict, vx, vy, wz):
    """Rim speeds (m/s, positive rolling forward) of a mecanum base for a chassis twist,
    from the interface file's `kinematics` row, in fl, fr, rl, rr order."""
    s = _lx_plus_ly(k)
    return [vx - vy - s * wz, vx + vy + s * wz, vx + vy - s * wz, vx - vy + s * wz]


def mecanum_ik(k: dict, vx, vy, wz):
    """Wheel speeds (rad/s) for a chassis twist (`mecanum_rim` over the wheel radius)."""
    r = float(k["wheel_radius"])
    return [v / r for v in mecanum_rim(k, vx, vy, wz)]


def mecanum_fk(k: dict, w):
    """The chassis twist (vx, vy, wz) of wheel speeds w (rad/s, fl, fr, rl, rr)."""
    r = float(k["wheel_radius"])
    fl, fr, rl, rr = w
    vx = r * (fl + fr + rl + rr) / 4.0
    vy = r * (-fl + fr + rl - rr) / 4.0
    wz = r * (-fl + fr - rl + rr) / (4.0 * _lx_plus_ly(k))
    return vx, vy, wz


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0
