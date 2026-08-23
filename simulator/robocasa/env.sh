# Environment for the RoboCasa simulator engine.
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
# The upstream clones, fetched by `run.sh setup`. The venv holds *editable*
# installs of both, so these directories must stay put.
export ROBOSUITE_DIR="$SIM_ROOT/upstream/robosuite"
export ROBOCASA_DIR="$SIM_ROOT/upstream/robocasa"
# Cross-engine resources (the wire bridge and robot spec files), shared by every
# simulator engine.
export SHARED_ROOT="$(realpath "$SIM_ROOT/../shared" 2>/dev/null || echo "$SIM_ROOT/../shared")"
export VENV_DIR="$SIM_ROOT/.venv"

# --- Rendering ---------------------------------------------------------------
# macOS has no EGL/OSMesa. glfw drives both the on-screen viewer and offscreen
# rendering (via a hidden window). Set MUJOCO_GL=cgl for pure headless offscreen.
if [ "$(uname -s)" = "Darwin" ]; then
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  # mjpython re-execs the interpreter and loses the default dyld fallback
  # path, so dlopen of @rpath-dependent libs (llvmlite's libllvmlite.dylib
  # needs @rpath/libz.1.dylib) fails inside the viewer process. Restore it.
  export DYLD_FALLBACK_LIBRARY_PATH="${DYLD_FALLBACK_LIBRARY_PATH:-/usr/lib:/usr/local/lib}"
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
fi

export PYTHONPATH="$ROBOCASA_DIR:$ROBOSUITE_DIR:$SIM_ROOT:$SHARED_ROOT:${PYTHONPATH:-}"

export TOKENIZERS_PARALLELISM=false
