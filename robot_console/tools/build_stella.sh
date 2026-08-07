#!/usr/bin/env bash
#
# Build the visual-SLAM engine's C++ half into robot_console/.vendor/.
#
#   tools/build_stella.sh              # build whatever is missing
#   tools/build_stella.sh --clean      # start from nothing
#
# Produces:
#   .vendor/src/                 checkouts of FBoW, g2o and stella_vslam
#   .vendor/stella/lib           libstella_vslam.dylib and its dependencies
#   .vendor/stella/share/orb_vocab.fbow   the ORB vocabulary (45 MB)
#   .venv-vslam/.../_stella*.so  the pybind11 binding the engine imports
#
# None of it is committed (see .gitignore): it is ~200 MB of build output that has to
# match the local compiler and python anyway. Run this once per machine -- including on
# a Jetson or any other box that will serve the engine over `--serve`.
#
# Everything installs into one prefix rather than onto the system, so nothing here can
# collide with a ROS or Homebrew copy of the same libraries.

set -euo pipefail

# Resolved without `cd`, for the same reason bin/*.sh avoid it: a shell whose chpwd
# hook writes a terminal-title escape would get that escape captured into the path.
SCRIPT="${BASH_SOURCE[0]}"
case "$SCRIPT" in
  /*) SCRIPT_PATH="$SCRIPT" ;;
  *)  SCRIPT_PATH="$PWD/$SCRIPT" ;;
esac
TOOLS_DIR="${SCRIPT_PATH%/*}"
CONSOLE_ROOT="${TOOLS_DIR%/*}"

VENDOR="$CONSOLE_ROOT/.vendor"
SRC="$VENDOR/src"
PREFIX="$VENDOR/stella"
VENV_VSLAM="${ROBOT_CONSOLE_VSLAM_VENV:-$CONSOLE_ROOT/.venv-vslam}"
PY="$VENV_VSLAM/bin/python"

# The upstream commits this was built and measured against. Pinned rather than
# tracking main: a tracker that silently changes its keyframe policy would move every
# number in tools/stella_offline.py's gate without anything in this repo changing.
FBOW_REPO=https://github.com/stella-cv/FBoW.git
FBOW_REF=c6e3c29
G2O_REPO=https://github.com/RainerKuemmerle/g2o.git
G2O_REF=e8df2004
STELLA_REPO=https://github.com/stella-cv/stella_vslam.git
STELLA_REF=8ac1be4
VOCAB_URL=https://github.com/stella-cv/FBoW_orb_vocab/raw/main/orb_vocab.fbow

if [ "${1:-}" = "--clean" ]; then
  echo "removing $VENDOR"
  rm -rf "$VENDOR"
fi

for tool in cmake git curl; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "error: $tool is required (brew install cmake git curl)" >&2
    exit 1
  }
done
if [ "$(uname)" = "Darwin" ]; then
  for formula in eigen opencv; do
    brew --prefix "$formula" >/dev/null 2>&1 || {
      echo "error: $formula is required (brew install $formula)" >&2
      exit 1
    }
  done
  # eigen@3/opencv@4 when they are the versioned formulae, plain otherwise.
  PREFIX_PATH="$PREFIX;$(brew --prefix eigen);$(brew --prefix opencv);$(brew --prefix)"
else
  PREFIX_PATH="$PREFIX"
fi

JOBS="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu || echo 4)"
mkdir -p "$SRC"

clone() {  # clone <dir> <repo> <ref>
  local dir="$SRC/$1" repo="$2" ref="$3"
  if [ ! -d "$dir/.git" ]; then
    echo "==> cloning $1"
    git clone "$repo" "$dir"
  fi
  git -C "$dir" fetch --quiet origin || true
  git -C "$dir" checkout --quiet "$ref"
  git -C "$dir" submodule update --init --recursive --quiet || true
}

build() {  # build <dir> <cmake args...>
  local dir="$SRC/$1"
  shift
  echo "==> building $dir"
  cmake -S "$dir" -B "$dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" \
    -DCMAKE_PREFIX_PATH="$PREFIX_PATH" \
    -DBUILD_SHARED_LIBS=ON \
    "$@"
  cmake --build "$dir/build" --config Release -j "$JOBS"
  cmake --install "$dir/build"
}

if [ ! -f "$PREFIX/lib/libfbow.dylib" ] && [ ! -f "$PREFIX/lib/libfbow.so" ]; then
  clone FBoW "$FBOW_REPO" "$FBOW_REF"
  # SSE off: this is built on Apple silicon as often as on x86, and the vocabulary
  # lookup is not where the time goes.
  build FBoW -DUSE_SSE=OFF -DUSE_SSE2=OFF -DUSE_SSE3=OFF -DUSE_SSE4=OFF
fi

if [ ! -f "$PREFIX/lib/libg2o_core.dylib" ] && [ ! -f "$PREFIX/lib/libg2o_core.so" ]; then
  clone g2o "$G2O_REPO" "$G2O_REF"
  # The vendored ceres and the LGPL csparse keep this to two Homebrew dependencies
  # (eigen, opencv); OpenMP is off because libomp is the one dependency that reliably
  # is not there on a fresh mac.
  build g2o \
    -DG2O_USE_CHOLMOD=OFF -DG2O_USE_CSPARSE=ON -DG2O_USE_LGPL_LIBS=ON \
    -DG2O_USE_OPENGL=OFF -DG2O_USE_OPENMP=OFF -DG2O_USE_VENDORED_CERES=ON \
    -DG2O_BUILD_APPS=OFF -DG2O_BUILD_EXAMPLES=OFF -DBUILD_UNITTESTS=OFF
fi

if [ ! -f "$PREFIX/lib/libstella_vslam.dylib" ] && [ ! -f "$PREFIX/lib/libstella_vslam.so" ]; then
  clone stella_vslam "$STELLA_REPO" "$STELLA_REF"
  build stella_vslam -DUSE_OPENMP=OFF -DUSE_SSE_ORB=OFF -DUSE_SSE_FP_MATH=OFF
fi

if [ ! -f "$PREFIX/share/orb_vocab.fbow" ]; then
  echo "==> fetching the ORB vocabulary (45 MB)"
  mkdir -p "$PREFIX/share"
  curl -fL "$VOCAB_URL" -o "$PREFIX/share/orb_vocab.fbow.part"
  mv "$PREFIX/share/orb_vocab.fbow.part" "$PREFIX/share/orb_vocab.fbow"
fi

# --- the python binding -------------------------------------------------------------
#
# Built against .venv-vslam's interpreter and installed into its site-packages, because
# that is the only python that ever imports it -- the console's own venv must stay at
# its three dependencies.
BINDING_SRC="$CONSOLE_ROOT/engines/stella"
if [ ! -f "$BINDING_SRC/CMakeLists.txt" ]; then
  cat >&2 <<EOF

The C++ half is built and installed in $PREFIX.

The pybind11 binding cannot be built: $BINDING_SRC/CMakeLists.txt is missing.
The engine imports \`_stella\`, so it needs that module; if this checkout has a
prebuilt one in .venv-vslam it will keep working, but nothing can regenerate it.
EOF
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "error: no $PY -- run bin/visual_slam.sh once to create .venv-vslam" >&2
  exit 1
fi
"$PY" -c "import pybind11" >/dev/null 2>&1 || {
  echo "error: pybind11 is missing from $VENV_VSLAM (uv pip install -e '.[vslam]')" >&2
  exit 1
}
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYBIND_CMAKE="$("$PY" -c 'import pybind11; print(pybind11.get_cmake_dir())')"
echo "==> building the _stella binding"
cmake -S "$BINDING_SRC" -B "$VENDOR/build-binding" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$SITE" \
  -DCMAKE_PREFIX_PATH="$PYBIND_CMAKE;$PREFIX_PATH" \
  -DPython_EXECUTABLE="$PY"
cmake --build "$VENDOR/build-binding" --config Release -j "$JOBS"
cmake --install "$VENDOR/build-binding"

"$PY" -c "import _stella; print('_stella ready:', _stella.__doc__)"
echo
echo "done. Vocabulary: $PREFIX/share/orb_vocab.fbow"
