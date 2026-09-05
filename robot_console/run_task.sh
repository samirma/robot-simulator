#!/usr/bin/env bash
# Run the apple-on-plate task against a running simulator, and grade it.
#
#   ./run_task.sh                          MolmoAct2 over ROS, one episode, ws://127.0.0.1:9090
#   ./run_task.sh --episodes 6             a pass count -- the only useful unit for a VLA
#   ./run_task.sh --label robocasa         name the run after the engine serving it
#   ./run_task.sh --instruction "..."      a different instruction (the scorers do not change)
#   ./run_task.sh --help
#
#   --url URL              rosbridge to connect to (default ws://127.0.0.1:9090)
#   --policy P             molmoact2 (default; the scripted so101_waypoint plan was removed)
#   --episodes N           episodes to run and count (default 1)
#   --steps N              per-episode step budget (default 400)
#   --robots A,B           robots expected on the wire, for the fleet check (default so101)
#   --namespace NS         the arm's ROS namespace (default so101)
#   --instruction TEXT     what the policy is told; default is the project's own text
#   --instruction-file F   the same, read from a file (long prompts do not survive quoting)
#   --log-dir DIR          where run logs go (default runs/task)
#   --label L              log subdirectory, since no engine is launched here (default: default)
#   --wait S               seconds to wait for the simulator's topics (default 180)
#   --reinstall            rebuild the venv this policy needs, then exit
#   --                     everything after this goes to inspect-robot
#
# The simulator is somebody else's job. Start one first, from simulator/:
#
#   ./kitchen.sh serve [--engine robocasa] [--viewer] [--wrist]
#
# and this script does the console's half: pick the venv the policy needs, wait for the
# *topics* (a listening socket says nothing about whether the scene compiled), check every
# expected robot is on the wire, then per episode reset-and-verify the world, run
# inspect-robot over the ros2_control contract, and grade the log with
# `robot_console.arm.verdict`. Each of those exists because its absence produced a
# confusing failure; the comments below say which.
#
# Report pass counts, never a single run. A VLA on this task is a coin toss even on the
# rig it was tuned for, and one episode of it tells you nothing. The scripted plan that
# used to serve as the transport check is gone; the preflight's IK reach gate and the
# fleet check are what now separate "the wire is broken" from "the policy missed".

set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
CONSOLE_ROOT="$(dirname "$_self")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }
usage() { awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$_self"; }

URL="ws://127.0.0.1:9090"
POLICY="molmoact2"
EPISODES=1
STEPS=400
ROBOTS="so101"
NS="so101"
INSTRUCTION=""
INSTRUCTION_FILE=""
LOG_DIR=""
LABEL="default"
WAIT=180
REINSTALL=0
declare -a PASSTHRU=()

# Parsed before anything is installed, so --help never bootstraps a venv.
while [ $# -gt 0 ]; do
  case "$1" in
    --url)              URL="$2"; shift 2 ;;
    --policy)           POLICY="$2"; shift 2 ;;
    --episodes)         EPISODES="$2"; shift 2 ;;
    --steps)            STEPS="$2"; shift 2 ;;
    --robots)           ROBOTS="$2"; shift 2 ;;
    --namespace)        NS="$2"; shift 2 ;;
    --instruction)      INSTRUCTION="$2"; shift 2 ;;
    --instruction-file) INSTRUCTION_FILE="$2"; shift 2 ;;
    --log-dir)          LOG_DIR="$2"; shift 2 ;;
    --label)            LABEL="$2"; shift 2 ;;
    --wait)             WAIT="$2"; shift 2 ;;
    --reinstall)        REINSTALL=1; shift ;;
    --)                 shift; PASSTHRU=("$@"); break ;;
    -h|--help|help)     usage; exit 0 ;;
    *) die "unknown flag '$1' (try: ./run_task.sh --help)" ;;
  esac
done
[ -z "$INSTRUCTION" ] || [ -z "$INSTRUCTION_FILE" ] \
  || die "give either --instruction or --instruction-file, not both"
[ -n "$LOG_DIR" ] || LOG_DIR="$CONSOLE_ROOT/runs/task"

# ---------------------------------------------------------------- the venv
#
# Two console venvs, and which one runs depends on the policy. torch is a multi-gigabyte
# dependency that only the VLA needs, and keeping it out of `.venv` is what makes the
# offline test suite fast and the scripted baseline cheap to run. Either venv carries the
# arm extra (`inspect-robots-ros`), so preflight, the fleet check, the episode and the
# verdict all run from the one chosen here.
case "$POLICY" in
  molmoact2) VENV_DIR="${ROBOT_CONSOLE_VLA_VENV:-$CONSOLE_ROOT/.venv-vla}"; EXTRA="vla" ;;
  *)         VENV_DIR="${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}";         EXTRA="dev,arm" ;;
esac
PY="$VENV_DIR/bin/python"
INSPECT="$VENV_DIR/bin/inspect-robot"
# A stamp of its own, not teleop.sh's: this install is a superset of teleop's (the arm
# extra on top of the base package), so it may satisfy teleop's stamp too, but a base-only
# install has no inspect-robot and must never satisfy this one.
STAMP="$VENV_DIR/.run-task-stamp"

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh"
  if [ ! -x "$PY" ]; then
    echo ">> creating venv ($VENV_DIR)"
    uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"
  fi
  echo ">> installing robot_console[$EXTRA] into $(basename "$VENV_DIR")"
  ( cd "$CONSOLE_ROOT" && VIRTUAL_ENV="$VENV_DIR" uv pip install -e ".[$EXTRA]" -q ) \
    || die "could not install robot_console[$EXTRA]"
  touch "$STAMP"
  [ "$VENV_DIR" != "${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}" ] || touch "$VENV_DIR/.teleop-stamp"
}

# Reinstall when the venv is missing or pyproject.toml has moved on. The editable install
# picks up source edits by itself; it is only the entry points -- the task, policies,
# embodiment and scorers this project registers with inspect-robots -- that need it, and
# a stale one does not fail loudly: the scorer name resolves in one venv and not the
# other, every episode dies before its first step, and the run prints `apple nan`.
if [ "$REINSTALL" -eq 1 ]; then
  bootstrap; exit 0
fi
if [ ! -x "$INSPECT" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi

# ---------------------------------------------------------------- the instruction
#
# The default is the project's own text (`robot_console.arm.task.INSTRUCTION`), read from
# the package rather than copied here so the two cannot drift. It is always passed
# explicitly: a run that relied on the task's default would record nothing about *why*
# it said what it said.
if [ -n "$INSTRUCTION_FILE" ]; then
  INSTRUCTION="$(<"$INSTRUCTION_FILE")"
fi
DEFAULT_INSTRUCTION="$("$PY" -c 'from robot_console.arm.task import INSTRUCTION; print(INSTRUCTION)')"
[ -n "$INSTRUCTION" ] || INSTRUCTION="$DEFAULT_INSTRUCTION"
say "instruction: $INSTRUCTION"
if [ "$INSTRUCTION" != "$DEFAULT_INSTRUCTION" ]; then
  echo "  custom text: the scorers still measure apple-on-plate geometry, so a 0 here says the" >&2
  echo "  apple did not end up on the plate -- not whether the policy did what it was told." >&2
fi

# ---------------------------------------------------------------- readiness
host_port="${URL#*://}"; host_port="${host_port%%/*}"
HOST="${host_port%%:*}"; PORT_NUM="${host_port##*:}"
[ "$HOST" != "$PORT_NUM" ] || PORT_NUM=9090

# Wait for the topics, not for the port. A listening socket only means the websocket
# server bound; it says nothing about whether the scene compiled, the cameras resolved,
# or the task staged. Every one of those failures otherwise arrives minutes later as a
# client-side reset timeout with no clue in it.
wait_for_topics() {
  local i said_closed=0 said_open=0
  for ((i = 1; i <= WAIT; i++)); do
    if ! nc -z "$HOST" "$PORT_NUM" 2>/dev/null; then
      if [ "$said_closed" -eq 0 ]; then
        echo "nothing listening on $URL - in another terminal, from simulator/:" >&2
        echo "  ./kitchen.sh serve [--engine robocasa] [--viewer] [--wrist]" >&2
        said_closed=1
      fi
    elif "$PY" -m robot_console.arm.preflight --url "$URL" --namespace "$NS" \
           --no-reset --allow-out-of-reach >/dev/null 2>&1; then
      echo " ready (${i}s)" >&2
      return 0
    elif [ "$said_open" -eq 0 ]; then
      # An open port is not a running simulator: Docker Desktop's proxy answers on 9090
      # with nothing listening behind it, and an engine that is still compiling its
      # kitchen has bound the socket long before it has anything to say.
      echo "port open, waiting for topics on $URL (an engine compiling, or a port held by" >&2
      echo "  something else - Docker Desktop's proxy does this on 9090; check docker ps)" >&2
      said_open=1
    fi
    [ $((i % 5)) -eq 0 ] && printf '.' >&2
    sleep 1
  done
  echo >&2
  die "no simulator published its topics on $URL within ${WAIT}s"
}

# Every robot in --robots is on the wire, under its own namespace. The preflight above
# only ever asks about the arm, and an arm episode can run perfectly while a second
# robot on the same port published nothing at all -- nothing the arm does touches the
# base's topics. So the multi-robot claim needs its own check, against the console's own
# contract constants rather than a list typed out here, which would drift from both sides.
check_fleet() {
  local expect=() name
  for name in ${ROBOTS//,/ }; do
    case "$name" in
      "$NS")  expect+=(--arm "$NS") ;;
      so101)  expect+=(--arm so101) ;;
      *)      expect+=(--base "$name") ;;
    esac
  done
  "$PY" -m robot_console.fleet --url "$URL" "${expect[@]}" \
    || die "the simulator is up but does not present every robot in --robots $ROBOTS"
}

say "== $LABEL: $POLICY on $URL"
printf 'waiting for topics' >&2
wait_for_topics
check_fleet

# ---------------------------------------------------------------- the episodes
passes=0
errors=0
for episode in $(seq 1 "$EPISODES"); do
  run_dir="$LOG_DIR/$LABEL/$(date +%Y%m%d-%H%M%S)-$episode"
  mkdir -p "$run_dir"

  # The preflight is separate from the episode's own reset for a reason: /reset says the
  # world was restored, and this checks that it *was* -- the apple measured back at a
  # spawn, and the grasp and release poses solving from there. A drifted apple otherwise
  # scores zero looking exactly like a policy failure.
  set +e
  "$PY" -m robot_console.arm.preflight --url "$URL" --namespace "$NS" \
    --json "$run_dir/scene_reset.json"
  preflight=$?
  set -e
  case "$preflight" in
    0) ;;
    2) die "no simulator answering on $URL" ;;
    3) die "the world did not reset - restart the engine" ;;
    4) die "the apple is somewhere the arm cannot reach, or at no layout's spawn" ;;
    *) die "preflight failed (exit $preflight)" ;;
  esac
  # Which layout the world is in -- standard, or the plate and apple swapped -- is read
  # off the wire by the preflight and handed to the task here, so the pose-derived
  # reference column grades the arrangement that exists. Typed in, it could disagree.
  LAYOUT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("layout") or "standard")' \
    "$run_dir/scene_reset.json")"
  [ "$episode" -ne 1 ] || echo "  layout: $LAYOUT"

  # fresh_obs_timeout_s: how long a step waits for an observation newer than the command
  # it just sent. The default is 2/control_hz = 0.2 s, which assumes the simulator
  # publishes faster than the control rate -- and this one does not: it renders its
  # cameras inside the physics loop and manages 8-10 Hz, and a VLA adds seconds of
  # inference on top. Measured: an episode died mid-run with "EmbodimentFault: no
  # post-publish joint state within fresh_obs_timeout_s=0.2s". Two seconds is generous,
  # and it is still a freshness guarantee -- a stale observation is refused, it is just
  # given time to arrive.
  #
  # --max-action-delta 0.65 raises the framework's own per-step limiter above the
  # policy's largest intended single-step change (the jaw closing 1.0 -> 0.40). The
  # default is derived from the action space and lands near 0.03, which halves the arm's
  # 0.06 rad step: measured, `home` and `pre_grasp` then burn their whole 40-step budget
  # without arriving and the episode runs out of steps in transit. The policy is already
  # the rate limiter, and it is the one holding the measured constants.
  #
  # --no-live-log: the framework's live snapshot is a second *.json in the run directory
  # whose status is `started` until the end; grading must never read it, and not writing
  # it is simpler than telling every reader to skip it.
  set +e
  "$INSPECT" run \
    --task apple_on_plate --policy "$POLICY" --embodiment so101_ros \
    -E "url=$URL" -E "namespace=$NS" -E "fresh_obs_timeout_s=2.0" \
    -T "max_steps=$STEPS" -T "instruction=$INSTRUCTION" -T "layout=$LAYOUT" \
    --max-action-delta 0.65 --grader none --no-prompt --no-live-log \
    --log-dir "$run_dir" ${PASSTHRU[@]+"${PASSTHRU[@]}"} > "$run_dir/episode.log" 2>&1
  status=$?
  set -e

  verdict="$("$PY" -m robot_console.arm.verdict "$run_dir")"
  read -r scored ref_said detail <<<"$verdict"
  if [ "$scored" = "PASS" ] && [ "$status" -eq 0 ]; then
    passes=$((passes + 1))
    printf '  episode %s/%s: \033[32mPASS\033[0m  apple %s' "$episode" "$EPISODES" "$detail"
  elif [ "$scored" = "ERROR" ] || [ "$status" -ne 0 ]; then
    errors=$((errors + 1))
    printf '  episode %s/%s: \033[31mERROR\033[0m %s (inspect-robot exit %s)' \
      "$episode" "$EPISODES" "$detail" "$status"
  else
    printf '  episode %s/%s: \033[31mFAIL\033[0m  apple %s (gate 0.080)' \
      "$episode" "$EPISODES" "$detail"
  fi
  # The graded verdict is the camera's. `reference_success` recomputes the same predicate
  # from the free-joint poses and grades nothing; it is printed only when the two
  # disagree, which is the one signal that separates "the policy failed" from "the
  # detector stopped seeing". A run where this fires repeatedly is a reason to look at the
  # overhead frames, not at the policy.
  [ "$ref_said" = "$scored" ] || [ "$scored" = "ERROR" ] \
    || printf '  \033[33m[camera says %s, pose reference says %s]\033[0m' "$scored" "$ref_said"
  echo "   log: $run_dir"

  if [ "$episode" -eq 1 ] && [ "$scored" != "ERROR" ]; then
    recorded="$("$PY" -m robot_console.arm.verdict --field instruction "$run_dir")"
    [ -z "$recorded" ] || [ "$recorded" = "$INSTRUCTION" ] \
      || echo "  warning: the log recorded a different instruction: $recorded" >&2
  fi
done

say "== $LABEL ($POLICY on $URL): $passes/$EPISODES passed, $errors errored"
[ "$passes" -gt 0 ]
