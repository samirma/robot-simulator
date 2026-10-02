# Environment for the MolmoSpaces engine. Sourced by run.sh.
#
# Resolved without cd: an interactive shell may have a chpwd/precmd hook that writes a
# terminal-title escape sequence to stdout, and `$(cd ... && pwd)` would capture it.
_env_src="${BASH_SOURCE[0]:-$0}"
case "$_env_src" in
  /*) ;;
  *) _env_src="$PWD/$_env_src" ;;
esac
ENGINE_ROOT="$(dirname "$_env_src")"
ENGINE_ROOT="$(realpath "$ENGINE_ROOT" 2>/dev/null || echo "${ENGINE_ROOT%/.}")"
unset _env_src
export ENGINE_ROOT
export ENGINE_NAME=molmospaces
# The upstream allenai/molmospaces clone, fetched by `run.sh setup` at a pinned revision.
export MOLMOSPACES_DIR="$ENGINE_ROOT/upstream"
export MOLMOSPACES_REV="713fd12ab593c3bbb4abfaa76622e14249aa36b3"
export SIMULATOR_ROOT="$(realpath "$ENGINE_ROOT/.." 2>/dev/null || echo "$ENGINE_ROOT/..")"
export SHARED_ROOT="$SIMULATOR_ROOT/shared"
export VENV_DIR="$ENGINE_ROOT/.venv"

# --- Asset storage -----------------------------------------------------------
# DATA_ROOT is what scripts/assets/hf_download.py extracts into; it creates a "mujoco/"
# subdirectory, which is exactly what MLSPACES_CACHE_DIR must point at.
export DATA_ROOT="$ENGINE_ROOT/data"
export MLSPACES_CACHE_DIR="$DATA_ROOT/mujoco"
export MLSPACES_ASSETS_DIR="$ENGINE_ROOT/assets"
export MLSPACES_FORCE_INSTALL=True

# --- Rendering ---------------------------------------------------------------
# macOS has no EGL/OSMesa. CGL renders offscreen from any thread without a window; the
# MuJoCo window (mjpython + mujoco.viewer) brings its own GLFW context.
if [ "$(uname -s)" = "Darwin" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-cgl}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

export PYTHONPATH="$MOLMOSPACES_DIR:$ENGINE_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p "$DATA_ROOT" "$MLSPACES_ASSETS_DIR"
