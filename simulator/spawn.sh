#!/usr/bin/env bash
# Spawn one robot by id into the running simulation and serve its vendor interface on its
# own rosbridge websocket (one per component for a composite robot).
#
#   ./spawn.sh <id> [--placement worktop|floor] [--sim-port <p>] [--port <p>] [--arm-port <p>]
#
# Runs in the foreground: it prints one readiness line naming each wire and its port once
# every wire serves its recorded interface, and Ctrl-C removes the robot and its wires.
# `./spawn.sh --help` lists every accepted robot id with its name.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

# Any set-up engine's venv runs the spawn client (it needs PyYAML for the interface
# files); a moved checkout is repaired first.
PY=""
for engine in molmospaces robocasa; do
  if [ -x "$ROOT/$engine/.venv/bin/python" ]; then
    python3 "$ROOT/shared/tools/relocate_venv.py" "$ROOT/$engine/.venv" || true
    PY="$ROOT/$engine/.venv/bin/python"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "error: no engine is set up; run simulator/molmospaces/run.sh setup (or robocasa)" >&2
  exit 2
fi
exec "$PY" -u "$ROOT/shared/spawn.py" "$@"
