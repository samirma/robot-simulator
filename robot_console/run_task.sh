#!/usr/bin/env bash
# Run the SO-101 apple_on_plate task with the MolmoAct2 VLA against a running simulator,
# and grade every episode with the camera-verdict scorer.
#
#   run_task.sh [--episodes N] [--label <engine>] [--url ws://...] [--robot <id>]
#               [--namespace <ns>] [--instruction <text> | --instruction-file <f>]
#               [-- <inspect-robot args>]
#
#   --episodes N           episodes to run and count (default 1; one episode is a smoke
#                          run, not a result)
#   --label L              the engine serving the wire, for the report; logs go to
#                          runs/task/<label>/, one run directory per episode
#   --url URL              rosbridge to connect to (default ws://127.0.0.1:9090)
#   --robot ID             a simulated robot id: the member expected besides the SO-101
#                          and the rig (default so101: the SO-101 alone). The SO-101 is
#                          always required; an id not listed below is refused
#   --namespace NS         the SO-101's namespace (default: discovered; '' is bare)
#   --instruction TEXT     what the policy is told (default: the task's own text)
#   --instruction-file F   the same, read from a file
#   --                     everything after this goes to inspect-robot unchanged
#
# The simulator is somebody else's job; start one first, from simulator/:
#
#   ./kitchen.sh serve [--engine robocasa]
#
# This refuses a wire without the simulation-only /reset or the worktop rig, so the arm
# task never runs on hardware. Report pass counts over many episodes, never one run.

set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
CONSOLE_ROOT="$(dirname "$_self")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }
usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$_self"; }

# The accepted --robot ids, each with its name, come from the console's copy of the
# simulated entries of robots_specs/robots.yml (the workspace parity tests hold the two
# equal). Standard library only, so --help and a refused id need no venv.
robot_ids() { PYTHONPATH="$CONSOLE_ROOT/src" python3 -m robot_console.robot_ids "$@"; }

#: An episode ends when the policy says it is done, or after this many policy steps.
MAX_POLICY_STEPS=220
#: How long to wait for the members' topics before giving up, seconds.
WAIT_S=180

URL="ws://127.0.0.1:9090"
EPISODES=1
ROBOT="so101"
NS=""
NS_GIVEN=0
INSTRUCTION=""
INSTRUCTION_FILE=""
LABEL="default"
declare -a PASSTHRU=()

# Parsed before anything is installed, so --help never bootstraps a venv.
while [ $# -gt 0 ]; do
  case "$1" in
    --url)              URL="$2"; shift 2 ;;
    --episodes)         EPISODES="$2"; shift 2 ;;
    --robot)            ROBOT="$2"; shift 2 ;;
    --namespace)        NS="$2"; NS_GIVEN=1; shift 2 ;;
    --instruction)      INSTRUCTION="$2"; shift 2 ;;
    --instruction-file) INSTRUCTION_FILE="$2"; shift 2 ;;
    --label)            LABEL="$2"; shift 2 ;;
    --)                 shift; PASSTHRU=("$@"); break ;;
    -h|--help|help)     usage; echo; echo "--robot accepts:"; robot_ids; exit 0 ;;
    *) die "unknown flag '$1' (try: ./run_task.sh --help)" ;;
  esac
done
[ -z "$INSTRUCTION" ] || [ -z "$INSTRUCTION_FILE" ] \
  || die "give either --instruction or --instruction-file, not both"
case "$EPISODES" in ''|*[!0-9]*|0) die "--episodes takes a positive integer" ;; esac
robot_ids check "$ROBOT" || exit 1
LOG_DIR="$CONSOLE_ROOT/runs/task/$LABEL"

# ---------------------------------------------------------------- 1. the venv
#
# The policy is the VLA, so the venv is `.venv-vla`: the base plus the `arm` and `vla`
# extras. torch lives only there, which keeps `.venv` and the offline suite torch-free.
VENV_DIR="${ROBOT_CONSOLE_VLA_VENV:-$CONSOLE_ROOT/.venv-vla}"
PY="$VENV_DIR/bin/python"
INSPECT="$VENV_DIR/bin/inspect-robot"
STAMP="$VENV_DIR/.run-task-stamp"

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh"
  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console[arm,vla] into $(basename "$VENV_DIR")"
  ( cd "$CONSOLE_ROOT" && VIRTUAL_ENV="$VENV_DIR" uv pip install -e ".[arm,vla]" -q ) \
    || die "could not install robot_console[arm,vla]"
  touch "$STAMP"
}

# A moved checkout leaves the venv's absolute paths in every shebang; re-point them.
[ ! -x "$PY" ] || "$PY" "$CONSOLE_ROOT/tools/relocate_venv.py" "$VENV_DIR" \
  || die "could not re-point $VENV_DIR at this checkout"
# Reinstall when the venv is missing or pyproject.toml moved on: the entry points --
# task, policy, embodiment, scorer -- are only registered by an install.
if [ ! -x "$INSPECT" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

# ---------------------------------------------------------------- the instruction
if [ -n "$INSTRUCTION_FILE" ]; then
  INSTRUCTION="$(<"$INSTRUCTION_FILE")"
fi
DEFAULT_INSTRUCTION="$("$PY" -c 'from robot_console.arm.task import INSTRUCTION; print(INSTRUCTION)')"
[ -n "$INSTRUCTION" ] || INSTRUCTION="$DEFAULT_INSTRUCTION"
say "instruction: $INSTRUCTION"
[ "$INSTRUCTION" = "$DEFAULT_INSTRUCTION" ] || echo "  custom text: the scorer still measures" \
  "apple-on-plate, not whether the policy did what it was told." >&2

preflight() { "$PY" -m robot_console.arm.preflight "$@" --url "$URL"; }

# The SO-101 and the rig are checked type for type by the arm preflight. Any other --robot
# is checked against its whole typed contract by the fleet check.
others_present() {
  [ "$ROBOT" = so101 ] || "$PY" -m robot_console.fleet --url "$URL" --expect "$ROBOT"
}

# ---------------------------------------------------------------- 2. the topics
host_port="${URL#*://}"; host_port="${host_port%%/*}"
HOST="${host_port%%:*}"; PORT_NUM="${host_port##*:}"
[ "$HOST" != "$PORT_NUM" ] || PORT_NUM=9090

say "== $LABEL: molmoact2 on $URL"
printf 'waiting for topics' >&2
said=0
ready=0
last=""
for ((i = 1; i <= WAIT_S; i += 5)); do
  if ! nc -z "$HOST" "$PORT_NUM" 2>/dev/null; then
    [ "$said" -eq 1 ] || echo " - nothing listening on $URL; from simulator/: ./kitchen.sh serve" >&2
    said=1; sleep 5; continue
  fi
  if [ "$NS_GIVEN" -eq 0 ]; then
    NS="$(preflight discover 2>/dev/null)" || {
      last="$(preflight discover 2>&1 >/dev/null)"; printf '.' >&2; sleep 5; continue; }
  fi
  # A wire without /reset or the rig is refused at once, by name, rather than waited on.
  set +e
  last="$(preflight check --namespace "$NS" 2>&1 >/dev/null)"
  refused=$?
  set -e
  if [ "$refused" -eq 5 ]; then
    echo >&2; echo "$last" >&2
    die "this wire is not a simulator serving the task"
  fi
  if preflight wait --namespace "$NS" --timeout 5 >/dev/null 2>&1 \
      && others_present >/dev/null 2>&1; then
    ready=1; break
  fi
  printf '.' >&2
done
echo >&2
if [ "$ready" -ne 1 ]; then
  [ -z "$last" ] || echo "$last" >&2
  die "no simulator published the members' topics on $URL within ${WAIT_S}s"
fi
if [ -n "$NS" ]; then echo "SO-101 on /$NS/*"; else echo "SO-101 on the bare contract"; fi

# ---------------------------------------------------------------- 3. the interfaces
set +e
preflight check --namespace "$NS"
check=$?
set -e
case "$check" in
  0) ;;
  5) die "this wire is not a simulator serving the task (see above)" ;;
  *) die "the SO-101's interface on $URL is not the one the task needs (exit $check)" ;;
esac
others_present || die "the wire does not present the --robot $ROBOT member"

# ---------------------------------------------------------------- 4. the episodes
[ "$EPISODES" -ne 1 ] || say "smoke run: one episode checks the pipeline; it is not a result"
passes=0
errors=0
completed=0
for episode in $(seq 1 "$EPISODES"); do
  run_dir="$LOG_DIR/$(date +%Y%m%d-%H%M%S)-$episode"
  mkdir -p "$run_dir"

  set +e
  reset_out="$(preflight reset --namespace "$NS" 2>&1)"
  reset_status=$?
  set -e
  if [ "$reset_status" -ne 0 ]; then
    echo "$reset_out" >&2
    say "== $LABEL: aborted before episode $episode; $completed episode(s) completed," \
      "$passes passed"
    exit 1
  fi

  # --max-action-delta 0.65 lifts the framework's per-step limiter above the policy's
  # largest intended single-step change (the jaw closing 1.0 -> 0.40); its default is
  # near 0.03 and halves the arm's step. fresh_obs_timeout_s gives joint state that is
  # newer than the command time to arrive when the VLA thinks for seconds between steps.
  set +e
  "$INSPECT" run \
    --task apple_on_plate --policy molmoact2 --embodiment so101_ros \
    -E "url=$URL" -E "namespace=$NS" -E "fresh_obs_timeout_s=2.0" \
    -T "max_steps=$MAX_POLICY_STEPS" -T "instruction=$INSTRUCTION" \
    --max-action-delta 0.65 --grader none --no-prompt --no-live-log \
    --log-dir "$run_dir" ${PASSTHRU[@]+"${PASSTHRU[@]}"} > "$run_dir/episode.log" 2>&1
  status=$?
  set -e
  # The embodiment calls /reset again as the framework starts the episode; a refusal
  # there aborts the run as the one above does.
  if refusal="$(grep -m1 -E '/reset refused' "$run_dir/episode.log")"; then
    echo "$refusal" >&2
    say "== $LABEL: aborted in episode $episode; $completed episode(s) completed, $passes passed"
    exit 1
  fi
  completed=$((completed + 1))

  read -r outcome detail <<<"$("$PY" -m robot_console.arm.verdict "$run_dir")"
  if [ "$outcome" = "PASS" ] && [ "$status" -eq 0 ]; then
    passes=$((passes + 1))
    printf '  episode %s/%s: \033[32mPASS\033[0m  %s' "$episode" "$EPISODES" "$detail"
  elif [ "$outcome" = "ERROR" ] || [ "$status" -ne 0 ]; then
    errors=$((errors + 1))
    printf '  episode %s/%s: \033[31mERROR\033[0m %s (inspect-robot exit %s)' \
      "$episode" "$EPISODES" "$detail" "$status"
  else
    printf '  episode %s/%s: \033[31mFAIL\033[0m  %s' "$episode" "$EPISODES" "$detail"
  fi
  echo "   log: $run_dir"
done

if [ "$EPISODES" -eq 1 ]; then
  say "== $LABEL smoke run (molmoact2 on $URL): $passes/1 passed, $errors errored -- not a result"
else
  say "== $LABEL (molmoact2 on $URL): $passes/$EPISODES passed, $errors errored"
fi
[ "$passes" -gt 0 ]
