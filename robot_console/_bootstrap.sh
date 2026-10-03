# Sourced by the console's launchers (teleop.sh, view.sh, python.sh); not run directly.
#
# Self-installs the console's venv on first run and again whenever pyproject.toml changes
# (its SHA-256 is kept in the venv; a mismatch reinstalls). Uses uv when available, else
# python3 -m venv + pip. Sets CONSOLE_PY to the venv's python.
#
#   ROBOT_CONSOLE_EXTRAS extras to install (default: dev, so the tests run from the same venv)

_rc_die() { echo "error: $*" >&2; exit 2; }

_rc_sha() {
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else sha256sum "$1" | cut -d' ' -f1; fi
}

CONSOLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
VENV_DIR="$CONSOLE_ROOT/.venv"
CONSOLE_PY="$VENV_DIR/bin/python"
_rc_stamp="$VENV_DIR/.pyproject.sha256"
_rc_extras="${ROBOT_CONSOLE_EXTRAS-dev}"
_rc_want="$(_rc_sha "$CONSOLE_ROOT/pyproject.toml")"

_rc_install() {
  local spec="$CONSOLE_ROOT"
  [ -n "$_rc_extras" ] && spec="$CONSOLE_ROOT[$_rc_extras]"
  echo ">> robot_console: installing venv at $VENV_DIR (pyproject.toml new or changed)" >&2
  if command -v uv >/dev/null 2>&1; then
    # 3.12: every dependency (pygame in particular) ships wheels for it.
    [ -x "$CONSOLE_PY" ] || uv venv -q --python 3.12 "$VENV_DIR" >&2 \
      || uv venv -q "$VENV_DIR" >&2 || _rc_die "uv venv failed"
    VIRTUAL_ENV="$VENV_DIR" uv pip install -q -e "$spec" >&2 || _rc_die "installing robot_console failed"
  else
    command -v python3 >/dev/null 2>&1 || _rc_die "neither uv nor python3 found"
    [ -x "$CONSOLE_PY" ] || python3 -m venv "$VENV_DIR" >&2 || _rc_die "python3 -m venv failed"
    "$CONSOLE_PY" -m pip install -q -e "$spec" >&2 || _rc_die "installing robot_console failed"
  fi
  echo "$_rc_want" > "$_rc_stamp"
}

if [ ! -x "$CONSOLE_PY" ] || [ ! -f "$_rc_stamp" ] || [ "$(cat "$_rc_stamp")" != "$_rc_want" ]; then
  _rc_install
fi
