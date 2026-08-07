#!/usr/bin/env bash
# Mapping and navigation from the robot's camera alone -- no lidar.
#
#   ./bin/visual_slam.sh map      --out runs/vhouse    drive by hand, map builds live
#   ./bin/visual_slam.sh navigate --map runs/vhouse    click a point, robot drives there
#
#   ./bin/visual_slam.sh map --robot ainex             the walking robot (no odometry)
#   ./bin/visual_slam.sh map --engine-url jetson:5555  engine on another machine
#   ./bin/visual_slam.sh <mode> --help                 every flag for that mode
#
# Keys (the map window must have focus):
#   W/S forward-back   A/D strafe   Q/E rotate   Space stop/pause
#   M save the map     Esc quit
#   navigate: left click sets a goal, right click cancels
#
# The tracker is stella_vslam, and it runs in a *second* venv (.venv-vslam) as a
# separate process: the console itself must stay installable with three pure-python
# dependencies. This script creates both venvs and points the console at the right
# interpreter through $ROBOT_CONSOLE_VSLAM_PY.
#
# Before the first run, build the C++ half once per machine:
#
#   tools/build_stella.sh
#
# There is deliberately no `explore` here, though `bin/slam.sh` has one. Monocular SLAM
# needs texture and *translation*: it initializes by triangulating between two views, so
# rotating in place tells it nothing, and the autonomous look-around that seeds frontier
# exploration is exactly that rotation. Six live runs all ended the same way -- tracker
# lost mid-spin, unable to relocalize, map stuck at nothing. Drive it yourself with
# `map`, keeping to open floor, and the tracker holds.
#
# The 3D window (rerun) opens alongside the map and camera windows and shows the
# tracker's own map -- every landmark and keyframe it holds -- as it grows. `--no-3d-view`
# turns it off.
set -euo pipefail

# Resolved without cd, so `--out runs/house` still means the caller's cwd. Using
# $(cd .. && pwd) here would also capture any terminal-title escapes the shell emits.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

VENV_DIR="${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}"
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.teleop-stamp"

VSLAM_VENV="${ROBOT_CONSOLE_VSLAM_VENV:-$CONSOLE_ROOT/.venv-vslam}"
VSLAM_PY="$VSLAM_VENV/bin/python"
VSLAM_STAMP="$VSLAM_VENV/.vslam-stamp"

VOCAB="$CONSOLE_ROOT/.vendor/stella/share/orb_vocab.fbow"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$_self"
}

need_uv() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh"
}

# The console venv, shared with teleop.sh and slam.sh -- one venv, one stamp, so
# switching between launchers never triggers a reinstall.
bootstrap_console() {
  need_uv
  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$CONSOLE_ROOT"
  touch "$STAMP"
}

# The engine venv. Deliberately separate and deliberately small: numpy, opencv,
# pybind11 and rerun. The compiled `_stella` module is installed into it by
# tools/build_stella.sh, which is why that step cannot be folded in here.
bootstrap_engine() {
  need_uv
  if [ ! -x "$VSLAM_PY" ]; then
    echo ">> creating engine venv ($VSLAM_VENV)"
    uv venv --python 3.12 "$VSLAM_VENV" || uv venv "$VSLAM_VENV"
  fi
  echo ">> installing robot_console[vslam]"
  VIRTUAL_ENV="$VSLAM_VENV" uv pip install -e "$CONSOLE_ROOT[vslam]"
  touch "$VSLAM_STAMP"
}

if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap_console
fi

for arg in "$@"; do
  if [ "$arg" = "--reinstall" ]; then
    bootstrap_console
    bootstrap_engine
    exit 0
  fi
done

[ $# -ge 1 ] || { usage; exit 2; }

mode="$1"; shift
case "$mode" in
  map)      module="robot_console.vslam.mapping" ;;
  navigate) module="robot_console.vslam.navigate" ;;
  -h|--help|help) usage; exit 0 ;;
  explore)
    usage
    die "there is no visual 'explore': autonomous exploration turns in place to look
    around, and a monocular tracker cannot initialize or hold tracking through that.
    Drive it yourself with 'map', or use the lidar app's bin/slam.sh explore." ;;
  *) usage; die "unknown mode '$mode' (expected map or navigate)" ;;
esac

# A remote engine needs none of the local machinery -- that is the point of --engine-url
# (a Jetson doing the tracking while the console runs wherever the operator is), so the
# engine venv and the vocabulary are only required when the engine is spawned here.
remote=0
for arg in "$@"; do
  case "$arg" in --engine-url|--engine-url=*|--engine-python|--engine-python=*) remote=1 ;; esac
done

if [ "$remote" -eq 0 ]; then
  if [ ! -x "$VSLAM_PY" ] || [ ! -f "$VSLAM_STAMP" ] \
     || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$VSLAM_STAMP" ]; then
    bootstrap_engine
  fi
  "$VSLAM_PY" -c "import _stella" >/dev/null 2>&1 \
    || die "the _stella module is missing from $VSLAM_VENV; build it with
    tools/build_stella.sh"
  [ -f "$VOCAB" ] || die "no ORB vocabulary at $VOCAB; build it with
    tools/build_stella.sh"
  export ROBOT_CONSOLE_VSLAM_PY="$VSLAM_PY"
fi

# exec, so Ctrl-C reaches Python directly -- which is what stops a real robot, since it
# has no command watchdog of its own.
exec "$PY" -m "$module" "$@"
