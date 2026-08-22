#!/usr/bin/env bash
# RoboCasa (robosuite + MuJoCo) simulator launcher.
#
#   ./run.sh setup                     install venv + robosuite + robocasa + assets
#   ./run.sh view [--scene kitchen:1]  open a kitchen in the MuJoCo viewer
#                 [--robot so101]      ...with a shared robot spawned in it:
#                                      so101 | myagv
#                 [--ros-port 9090]    ...on the myAGV's vendor ROS topics (myagv only)
#                 [--control 127.0.0.1:8000] ...or as a generic control server (so101)
#                 [--headless]         ...run the sim + servers with no viewer window
#   ./run.sh shell                     interactive shell inside the venv
#
# The robot models and the wire bridge come from simulator/shared, so robot_console
# connects to a RoboCasa-hosted robot exactly as it does to a MolmoSpaces one.
#
# Any flags after the subcommand are forwarded to the underlying entry point.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
SIM_ROOT="$(dirname "$_self")"
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
# shellcheck source=/dev/null
source "$SIM_ROOT/env.sh"

PY="$VENV_DIR/bin/python"
# The MuJoCo passive viewer must own the main thread on macOS, which is what
# mjpython provides. Everything else runs under plain python.
MJPY="$VENV_DIR/bin/mjpython"
[ "$(uname -s)" = "Darwin" ] || MJPY="$PY"

die() { echo "error: $*" >&2; exit 1; }

find_python311() {
  for c in /opt/homebrew/opt/python@3.11/bin/python3.11 \
           /usr/local/opt/python@3.11/bin/python3.11 \
           "$(command -v python3.11 || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

# ---------------------------------------------------------------- setup

do_setup() {
  command -v uv >/dev/null || die "uv not found; install from https://docs.astral.sh/uv/"
  command -v git >/dev/null || die "git not found"

  if [ ! -x "$PY" ]; then
    # mjpython needs a shared libpython, which uv's standalone CPython does not ship,
    # so the venv is built on a Homebrew/system framework Python 3.11.
    py311="$(find_python311)" || die "python 3.11 not found. Run: brew install python@3.11"
    echo ">> creating venv on $py311"
    uv venv --python "$py311" "$VENV_DIR"
  fi

  mkdir -p "$UPSTREAM_DIR"

  # robosuite: the docs are explicit -- use the master branch, not a tag.
  if [ ! -d "$ROBOSUITE_DIR/.git" ]; then
    echo ">> cloning robosuite (master)"
    git clone https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_DIR"
  fi
  echo ">> installing robosuite"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ROBOSUITE_DIR"

  if [ ! -d "$ROBOCASA_DIR/.git" ]; then
    echo ">> cloning robocasa"
    git clone https://github.com/robocasa/robocasa.git "$ROBOCASA_DIR"
  fi
  echo ">> installing robocasa"
  # robocasa's full dependency set pulls heavy RL/learning packages (tianshou, lerobot)
  # that are irrelevant to hosting a kitchen scene and often fail to resolve on macOS
  # arm64. We only need the scene/arena + robosuite, so install without deps and add the
  # handful the models code actually imports.
  if ! VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ROBOCASA_DIR"; then
    echo ">> full robocasa install failed; retrying without heavy deps"
    VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$ROBOCASA_DIR" --no-deps
    VIRTUAL_ENV="$VENV_DIR" uv pip install \
      "mujoco==3.3.1" numpy scipy opencv-python Pillow imageio h5py \
      termcolor tqdm pyyaml
  fi

  # robocasa pins mujoco==3.3.1; make sure that is what we have.
  VIRTUAL_ENV="$VENV_DIR" uv pip install "mujoco==3.3.1"
  # The shared control server / camera path needs these.
  VIRTUAL_ENV="$VENV_DIR" uv pip install msgpack-numpy websockets

  echo ">> setting up robocasa macros"
  "$PY" -m robocasa.scripts.setup_macros || true

  echo ">> downloading kitchen assets (~10 GB)"
  # The downloader prompts "Proceed? (y/n)"; setup is non-interactive here.
  echo y | "$PY" -m robocasa.scripts.download_kitchen_assets

  echo ">> setup complete"
}

ensure_setup() {
  [ -x "$PY" ] || die "not installed yet - run: ./run.sh setup"
}

# ---------------------------------------------------------------- view

do_view() {
  ensure_setup
  # spawn_robot parses --robot/--scene/--ros-port/--control/--headless itself.
  local runner="$MJPY"
  # A headless run does not need the viewer's main thread, so plain python is fine
  # (and works on displayless hosts).
  case " $* " in *" --headless "*) runner="$PY" ;; esac
  exec "$runner" "$SIM_ROOT/tools/spawn_robot.py" "$@"
}

# ---------------------------------------------------------------- shell

do_shell() {
  ensure_setup
  echo ">> venv: $VENV_DIR   MUJOCO_GL=$MUJOCO_GL"
  exec "${SHELL:-/bin/bash}" -i
}

# ---------------------------------------------------------------- dispatch

cmd="${1:-help}"
[ $# -gt 0 ] && shift || true

case "$cmd" in
  setup)  do_setup "$@" ;;
  view)   do_view "$@" ;;
  shell)  do_shell "$@" ;;
  help|-h|--help)
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
    ;;
  *) die "unknown command '$cmd' (try: ./run.sh help)" ;;
esac
