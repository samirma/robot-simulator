#!/usr/bin/env bash
# A VLA model performs a task on the SO-101 arm in each simulator engine.
#
#   ./bin/arm_task.sh                          the default task in both simulators
#   ./bin/arm_task.sh "push the bowl left"     a custom task, positional
#   ./bin/arm_task.sh --task "..." --ports 8000    ...one simulator only
#   ./bin/arm_task.sh --dry-run                predict and print; the arm holds still
#   ./bin/arm_task.sh --device cpu --dtype float32   the MPS escape hatch
#   ./bin/arm_task.sh --help                   every flag
#
# Default task: "move the arm towards the apple, grasp it, lift it up, and drop it
# into the bowl" -- which matches what `simulator/kitchen_arm.sh serve` stages next to
# the arm in both engines (MolmoSpaces on port 8000, RoboCasa on 8001). Start that
# first; this connects to both in turn with one model load.
#
# The model is MolmoAct2-SO100_101 (Ai2, 5B) -- the 2026-08 survey's best open-weight
# model with both a checkpoint for this exact arm and a working PyTorch path on Apple
# Silicon. The wire protocol is molmospaces-control-v1, the same one robot-console-arm
# speaks: ROS was considered and rejected because the simulators' rosbridge carries
# only the myAGV's vendor topics, and a real SO-101 does not speak ROS -- an engine
# that served the arm over ROS would be presenting an interface no hardware has.
#
# Run a --dry-run first when changing --joint-offsets/--gripper-mode: it prints the
# fed state next to the predicted chunk, and a correct mapping predicts near the
# current pose for a resting arm.
#
# The first run creates .venv-vla (torch lives there, never in .venv -- the offline
# test suite stays fast) and downloads ~22 GB of model weights from Hugging Face.
set -euo pipefail

# Resolved without cd; see the note in teleop.sh about terminal-title escapes.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

# Its own venv and its own stamp, deliberately not teleop.sh's: torch is a
# multi-gigabyte install, and sharing .venv would slow every teleop bootstrap and
# every pytest run for a dependency only this launcher needs.
VENV_DIR="${ROBOT_CONSOLE_VLA_VENV:-$CONSOLE_ROOT/.venv-vla}"
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.vla-stamp"

die() { echo "error: $*" >&2; exit 1; }

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh"

  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console[vla] (torch: this takes a while the first time)"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$CONSOLE_ROOT[vla]"
  touch "$STAMP"
}

if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

for arg in "$@"; do
  if [ "$arg" = "--reinstall" ]; then
    bootstrap
    exit 0
  fi
done

# An op the checkpoint needs that MPS has not implemented falls back to CPU instead of
# aborting the run mid-episode.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# exec, so Ctrl-C reaches Python directly and the ArmClient closes its socket -- the
# simulator holds the last commanded target, so a clean close is a parked arm.
exec "$PY" -m robot_console.arm_task "$@"
