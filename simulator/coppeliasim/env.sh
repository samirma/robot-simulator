# Environment for the CoppeliaSim simulator engine.
# Sourced by run.sh; can also be sourced directly into an interactive shell.
#
# Mirrors the other engines' env.sh: resolve this file's directory WITHOUT cd'ing,
# because an interactive shell may have a chpwd/precmd hook that writes a terminal-title
# escape sequence to stdout, which `$(cd ... && pwd)` would capture into the path.
_env_src="${BASH_SOURCE[0]:-$0}"
case "$_env_src" in
  /*) ;;
  *) _env_src="$PWD/$_env_src" ;;
esac
SIM_ROOT="$(dirname "$_env_src")"
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
unset _env_src
export SIM_ROOT

# CoppeliaSim EDU. macOS arm64 build; there is no stable "latest" URL, so the version is
# pinned here and can be bumped in one place. By downloading you accept the EDU terms at
# coppeliarobotics.com (educational use only).
export COPSIM_VER="${COPSIM_VER:-V4_9_0_rev6}"
export COPSIM_OS="${COPSIM_OS:-macOS15_arm64}"
export COPSIM_ZIP="CoppeliaSim_Edu_${COPSIM_VER}_${COPSIM_OS}.zip"
export COPSIM_URL="https://downloads.coppeliarobotics.com/${COPSIM_VER}/${COPSIM_ZIP}"

# The unpacked .app bundle and the executable inside it.
export APP_DIR="$SIM_ROOT/app"
export COPSIM_APP="$APP_DIR/CoppeliaSim.app"
export COPSIM_BIN="$COPSIM_APP/Contents/MacOS/coppeliaSim"
# Some clients resolve Resources/ through this.
export COPPELIASIM_ROOT_DIR="$COPSIM_APP/Contents/Resources"

# Cross-engine resources (the wire bridge and robot spec files).
export SHARED_ROOT="$(realpath "$SIM_ROOT/../shared" 2>/dev/null || echo "$SIM_ROOT/../shared")"
export VENV_DIR="$SIM_ROOT/.venv"

export DATA_ROOT="$SIM_ROOT/data"

# The adapter imports the shared wire bridge (contracts.*), the shared robot specs
# (robots_spec) and the shared MuJoCo-free helpers, plus this engine's own tools.
export PYTHONPATH="$SIM_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"

# The ZMQ remote API server CoppeliaSim starts on launch.
export COPSIM_ZMQ_PORT="${COPSIM_ZMQ_PORT:-23000}"

mkdir -p "$DATA_ROOT"
