# Environment for the MolmoSpaces simulator.
# Sourced by run.sh; can also be sourced directly into an interactive shell.

# Resolve this file's directory WITHOUT cd'ing: an interactive shell may have a
# chpwd/precmd hook that writes a terminal-title escape sequence to stdout, and
# `$(cd ... && pwd)` would capture that junk into the path.
_env_src="${BASH_SOURCE[0]:-$0}"
case "$_env_src" in
  /*) ;;
  *) _env_src="$PWD/$_env_src" ;;
esac
SIM_ROOT="$(dirname "$_env_src")"
# realpath is an external binary, so it also cannot trigger a shell hook.
SIM_ROOT="$(realpath "$SIM_ROOT" 2>/dev/null || echo "${SIM_ROOT%/.}")"
unset _env_src
export SIM_ROOT
# The upstream allenai/molmospaces clone, fetched by `run.sh setup`.
export MOLMOSPACES_DIR="$SIM_ROOT/upstream"
# Cross-engine resources (the wire bridge and robot spec files), shared by every
# simulator engine. Resolved without cd; realpath is an external binary.
export SHARED_ROOT="$(realpath "$SIM_ROOT/../shared" 2>/dev/null || echo "$SIM_ROOT/../shared")"
export VENV_DIR="$SIM_ROOT/.venv"

# --- Asset storage -----------------------------------------------------------
# DATA_ROOT is what scripts/assets/hf_download.py extracts into; it creates a
# "mujoco/" subdirectory, which is exactly what MLSPACES_CACHE_DIR must point at.
export DATA_ROOT="$SIM_ROOT/data"
export MLSPACES_CACHE_DIR="$DATA_ROOT/mujoco"
export MLSPACES_ASSETS_DIR="$SIM_ROOT/assets"
export MLSPACES_FORCE_INSTALL=True

# --- Rendering ---------------------------------------------------------------
# macOS has no EGL/OSMesa. glfw drives both the on-screen viewer and offscreen
# rendering (via a hidden window). Set MUJOCO_GL=cgl for pure headless offscreen.
if [ "$(uname -s)" = "Darwin" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

export PYTHONPATH="$MOLMOSPACES_DIR:$SIM_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"

# Quieter, more deterministic runs.
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$DATA_ROOT" "$MLSPACES_ASSETS_DIR"
