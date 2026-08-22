# Environment for the RoboCasa (robosuite + MuJoCo) simulator engine.
# Sourced by run.sh; can also be sourced directly into an interactive shell.
#
# Mirrors simulator/molmospaces/env.sh: resolve this file's directory WITHOUT cd'ing,
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

# Upstream clones, fetched by `run.sh setup`. They carry their own .git.
export UPSTREAM_DIR="$SIM_ROOT/upstream"
export ROBOSUITE_DIR="$UPSTREAM_DIR/robosuite"
export ROBOCASA_DIR="$UPSTREAM_DIR/robocasa"

# Cross-engine resources (the wire bridge and robot spec files).
export SHARED_ROOT="$(realpath "$SIM_ROOT/../shared" 2>/dev/null || echo "$SIM_ROOT/../shared")"
export VENV_DIR="$SIM_ROOT/.venv"

# RoboCasa downloads its kitchen assets into the robocasa package tree; keep scratch
# output here.
export DATA_ROOT="$SIM_ROOT/data"

# --- Rendering ---------------------------------------------------------------
# macOS has no EGL/OSMesa. glfw drives the on-screen viewer and offscreen rendering
# (via a hidden window); cgl is pure headless offscreen.
if [ "$(uname -s)" = "Darwin" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

# The engine adapters import the shared wire bridge (contracts.*) and robot specs
# (robots_spec), plus this engine's own robots/ adapter package.
export PYTHONPATH="$SIM_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$DATA_ROOT"
