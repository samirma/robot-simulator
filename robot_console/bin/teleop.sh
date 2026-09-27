#!/usr/bin/env bash
# Keyboard teleoperation of a myAGV or an AiNex, simulated or real.
#
#   teleop.sh [--robot <id>] [--namespace <ns>] [--url ws://…] [--record <dir>]
#             [--speed <m/s>] [--max-speed <m/s>] [--latch]
#             [--safety-timeout <s>] [--no-preflight] [--reinstall]
#
#   ./bin/teleop.sh                               drive whatever robot is on ws://127.0.0.1:9090
#   ./bin/teleop.sh --robot ainex                 ...an AiNex, which walks rather than rolls
#   ./bin/teleop.sh --namespace ''                ...the bare contract a real bringup presents
#   ./bin/teleop.sh --url ws://192.168.1.42:9090  ...a real myAGV on the network
#   ./bin/teleop.sh --record runs/drive1          ...writing feed.mp4 + commands.jsonl
#
# Motion is published only by a separate safety supervisor process, which stops the robot
# (its stop_command, three times) if this UI's heartbeat stops for --safety-timeout
# (0.25 s). Before any motion you are asked to confirm that an independent physical
# emergency stop is armed: that device, not software, covers host failure and network
# loss. --no-preflight and --reinstall skip neither.
#
# With neither --robot nor --namespace given, both are read off the wire: /rosapi/topics
# says which robots are on that rosbridge and what each one is called. It has to be asked,
# because the simulator names every robot after itself (/myagv/cmd_vel) while these
# constants are the bare vendor contract (/cmd_vel) -- and a client subscribed to the
# wrong one gets no error, just a black window and a robot that ignores every key.
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

# A moved or copied checkout leaves the venv's absolute path in every script's shebang
# and in the editable install, so `python` starts while `inspect-robot` and the package
# import do not. Re-pointed in place (a no-op when nothing moved) rather than rebuilt,
# which would re-resolve every dependency. See tools/relocate_venv.py.
[ ! -x "$PY" ] || "$PY" "$CONSOLE_ROOT/tools/relocate_venv.py" "$VENV_DIR" \
  || die "could not re-point $VENV_DIR at this checkout"

# Reinstall when the venv is missing or the dependencies have moved on, so a pull that
# changes pyproject.toml does not need a separate setup step.
if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

for arg in "$@"; do
  if [ "$arg" = "--reinstall" ]; then
    # Rebuild, then carry on: the flag is forwarded and ignored by Python, and it skips
    # neither the safety supervisor nor the emergency-stop confirmation.
    bootstrap
    break
  fi
done

# exec, so Ctrl-C reaches the UI directly; it asks the safety supervisor to stop the
# robot, which has no command watchdog of its own.
exec "$PY" -m robot_console "$@"
