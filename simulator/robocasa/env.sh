# Environment for the RoboCasa engine. Sourced by run.sh.
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
export ENGINE_NAME=robocasa
# The upstream clones, fetched by `run.sh setup` at pinned revisions. The venv holds
# editable installs of both, so these directories must stay put (run.sh repair follows
# a moved checkout).
export ROBOSUITE_DIR="$ENGINE_ROOT/upstream/robosuite"
export ROBOCASA_DIR="$ENGINE_ROOT/upstream/robocasa"
export ROBOSUITE_REV="5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
export ROBOCASA_REV="8f3c96ec8d1bfcd8126cad2bca887da98d30e997"   # tag v1.0
export SIMULATOR_ROOT="$(realpath "$ENGINE_ROOT/.." 2>/dev/null || echo "$ENGINE_ROOT/..")"
export SHARED_ROOT="$SIMULATOR_ROOT/shared"
export VENV_DIR="$ENGINE_ROOT/.venv"
# The generated kitchen asset tree (upstream's committed assets plus the downloaded
# sources). robocasa reads `robocasa.models.assets_root`, which the scene loader points
# here, so nothing is ever extracted into the upstream checkout.
export ROBOCASA_ASSETS_DIR="$ENGINE_ROOT/assets"

# --- Rendering ---------------------------------------------------------------
if [ "$(uname -s)" = "Darwin" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-cgl}"
  # mjpython re-execs the interpreter and loses the default dyld fallback path, so
  # dlopen of @rpath-dependent libs (llvmlite needs @rpath/libz.1.dylib) fails inside
  # the viewer process. Restore it.
  export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/usr/lib:/usr/local/lib}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

export PYTHONPATH="$ROBOCASA_DIR:$ROBOSUITE_DIR:$ENGINE_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
