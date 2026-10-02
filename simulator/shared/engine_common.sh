# Shared by simulator/<engine>/run.sh (sourced after the engine's env.sh). One copy of
# argument checking, checkouts, venv creation, repair and `start` for both engines.

PY="$VENV_DIR/bin/python"

die() { echo "error: $*" >&2; exit 1; }

usage() {
  # The header comment block of the calling run.sh is its help.
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
}

need_uv() {
  command -v uv >/dev/null || die "uv not found; install it from https://docs.astral.sh/uv/"
}

find_python311() {
  local c
  for c in /opt/homebrew/opt/python@3.11/bin/python3.11 \
           /usr/local/opt/python@3.11/bin/python3.11 \
           "$(command -v python3.11 || true)"; do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}

make_venv() {
  if [ ! -x "$PY" ]; then
    # mjpython needs a shared libpython, which uv's standalone CPython does not ship, so
    # the venv is built on a framework Python 3.11 (Homebrew's python@3.11 on macOS).
    local py311
    py311="$(find_python311)" || die "python 3.11 not found (macOS: brew install python@3.11)"
    echo ">> creating $VENV_DIR on $py311"
    uv venv -q --python "$py311" "$VENV_DIR"
  fi
}

# checkout <url> <revision> <dir>: clone once, at the pinned revision. An existing clone
# at another revision is moved to the pin only when it has no local changes: vendored
# upstreams are never modified.
checkout() {
  local url="$1" rev="$2" dir="$3" head
  if [ ! -d "$dir/.git" ]; then
    echo ">> cloning $url at $rev"
    mkdir -p "$(dirname "$dir")"
    git clone -q "$url" "$dir"
    git -C "$dir" -c advice.detachedHead=false checkout -q "$rev"
    return
  fi
  head="$(git -C "$dir" rev-parse HEAD)"
  if [ "$head" != "$rev" ]; then
    [ -z "$(git -C "$dir" status --porcelain --untracked-files=no)" ] \
      || die "$dir has local changes and is not at the pinned $rev; restore it first"
    echo ">> moving $dir to the pinned $rev"
    git -C "$dir" fetch -q origin "$rev" 2>/dev/null || git -C "$dir" fetch -q origin
    git -C "$dir" -c advice.detachedHead=false checkout -q "$rev"
  fi
}

# The robot meshes the robot specification's folders need but the repository does not
# commit: fetched from each robot's pinned sources and verified against
# robots_specs/meshes.sha256 by robots_specs' own tool, which refuses on a mismatch and
# leaves no unverified file behind.
fetch_robot_meshes() {
  local tool="$SIMULATOR_ROOT/../robots_specs/tools/fetch_meshes.py"
  [ -f "$tool" ] || die "robots_specs/tools/fetch_meshes.py is missing"
  echo ">> fetching and verifying the robot meshes (robots_specs/meshes.sha256)"
  python3 "$tool" || die "robot meshes could not be fetched and verified (see above)"
}

# The worktop objects' YCB meshes (simulator/shared/worktop_objects.py), which the
# repository does not commit: fetched from elpis-lab/YCB_Dataset at a pinned commit and
# verified against simulator/shared/objects/ycb.sha256; a mismatch is refused and leaves
# no unverified file behind.
fetch_worktop_objects() {
  echo ">> fetching and verifying the worktop objects (simulator/shared/objects/ycb.sha256)"
  python3 "$SHARED_ROOT/tools/fetch_objects.py" \
    || die "worktop objects could not be fetched and verified (see above)"
}

ensure_setup() {
  [ -x "$PY" ] || die "$ENGINE_NAME is not set up yet; run: $ENGINE_ROOT/run.sh setup"
  do_repair
}

# start [--scene <source>:<id>] [--sim-port <p>] [--mujoco]
start_simulation() {
  local scene="" port="9080" window=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --scene)
        [ $# -ge 2 ] || die "--scene needs a value <source>:<id>"
        scene="$2"; shift 2 ;;
      --scene=*) scene="${1#*=}"; shift ;;
      --sim-port)
        [ $# -ge 2 ] || die "--sim-port needs a port number"
        port="$2"; shift 2 ;;
      --sim-port=*) port="${1#*=}"; shift ;;
      --mujoco) window=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "start: unknown flag '$1' (see $ENGINE_ROOT/run.sh --help)" ;;
    esac
  done
  case "$port" in ''|*[!0-9]*) die "--sim-port: expected a port number, got '$port'" ;; esac
  [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || die "--sim-port: expected 1-65535, got $port"
  ensure_setup
  local py="$PY"
  declare -a extra=()
  if [ "$window" -eq 1 ]; then
    extra+=(--mujoco)
    # On macOS the MuJoCo viewer must own the main thread, which mjpython provides.
    [ "$(uname -s)" = Darwin ] && py="$VENV_DIR/bin/mjpython"
  fi
  [ -n "$scene" ] && extra+=(--scene "$scene")
  exec "$py" "$SHARED_ROOT/simulation.py" --engine "$ENGINE_NAME" --sim-port "$port" \
    ${extra[@]+"${extra[@]}"}
}
