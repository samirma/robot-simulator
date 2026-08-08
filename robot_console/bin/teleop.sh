#!/usr/bin/env bash
# Keyboard teleoperation for a mobile robot, simulated or real.
#
#   ./bin/teleop.sh                        drive a myAGV on ws://127.0.0.1:9090
#   ./bin/teleop.sh --robot ainex          ...an AiNex, which walks rather than rolls
#   ./bin/teleop.sh --host 192.168.1.42    ...a real myAGV on the network
#   ./bin/teleop.sh --record runs/drive1   ...writing feed.mp4 + commands.jsonl
#   ./bin/teleop.sh --no-preflight         skip the "is anything listening" check
#   ./bin/teleop.sh --help                 every flag
#
# --robot picks the speed envelope, the on-screen wording and the wire contract: the
# myAGV is driven by a Twist on /cmd_vel and reports /odom back; the AiNex has neither,
# and the same keys are turned into gait parameters instead. Each robot's speed limits
# are its own hardware's, so a drive rehearsed in the simulator matches.
#
# Keys (the camera window must have focus):
#   W/S forward-back   A/D strafe   Q/E rotate   Space stop   +/- speed   Esc quit
#   On the AiNex these walk, sidestep and turn -- same keys, different gait.
#
# The first run creates .venv and installs the package; after that this is just a
# launcher. Flags are forwarded to `python -m robot_console`.
set -euo pipefail

# Resolved without cd, so `--record runs/drive1` still means the caller's cwd. Using
# $(cd .. && pwd) here would also capture any terminal-title escapes the shell emits.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

VENV_DIR="${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}"
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.teleop-stamp"

die() { echo "error: $*" >&2; exit 1; }

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh
  or set the venv up by hand:
    python3 -m venv '$VENV_DIR' && '$VENV_DIR/bin/pip' install -e '$CONSOLE_ROOT'"

  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    # No mjpython/framework-Python constraint here -- unlike the simulator, the console
    # is happy on uv's standalone CPython.
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$CONSOLE_ROOT"
  touch "$STAMP"
}

# Reinstall when the venv is missing or the dependencies have moved on, so a pull that
# changes pyproject.toml does not need a separate setup step.
if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

for arg in "$@"; do
  if [ "$arg" = "--reinstall" ]; then
    bootstrap
    exit 0
  fi
done

# exec, so Ctrl-C reaches Python directly -- which is what stops a real AGV, since it
# has no command watchdog of its own.
exec "$PY" -m robot_console "$@"
