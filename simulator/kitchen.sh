#!/usr/bin/env bash
# The SO-101 on a kitchen work surface, in *both* engines, around the same two objects.
#
#   ./kitchen.sh                 render a screenshot from each engine into shots/
#   ./kitchen.sh view            open both engines in the MuJoCo viewer
#   ./kitchen.sh serve           run on rosbridge (see --engine); add --viewer for a window
#   ./kitchen.sh inspect         stage the task, run inspect-robot against it, PASS/FAIL
#   ./kitchen.sh cameras         open the live camera page on a running `serve`/`inspect`
#   ./kitchen.sh help
#
#   --objects bowl,apple   the pair to stage the scene around (default: bowl,apple)
#   --scene ithor:1        MolmoSpaces scene    (default: ithor:1, a kitchen)
#   --layout 1 --style 1   RoboCasa kitchen     (both 1-60)
#   --out DIR              where screenshots go (default: shots/)
#   --ports 9090,9091      rosbridge ports for `serve`/`inspect` (molmospaces, robocasa)
#
#   serve / inspect:
#   --policy P             so101_waypoint (default) | molmoact2
#   --engine E             molmospaces (default) | robocasa | both. `serve` and
#                          `inspect` need only the engines named here to be set up;
#                          `shot` and `view` are comparisons and need both.
#   --steps N              episode budget (default 400)
#   --episodes N           run N episodes and report the pass count (default 1)
#   --wrist                also stream the eye-in-hand camera
#   --viewer               open the engine's own MuJoCo window while it serves. One
#                          process, so the window shows exactly the state on the wire.
#                          It costs camera rate -- measured on MolmoSpaces with three
#                          cameras, 7.6 Hz headless against 5.0 Hz with the window --
#                          and the loop still holds real time at that price.
#   --no-reference-table   put the task's objects on the engine's own worktop instead
#                          of on the reference work surface
#   --no-dressing          apple and plate only, without the bowl/mug/banana/lemon
#   --reference-lighting   impose the reference rig's exposure on the kitchen. Off by
#                          default, against what the photometry says: it matches the
#                          reference's clipped-pixel fraction almost exactly and costs
#                          MolmoAct2 the task outright (0/24 episodes with it, 6/18
#                          without). For comparing exposure, not for running a policy.
#   --extra-lights         add the reference's two lamps (they blow out a lit kitchen)
#   --http-port 8791       port the `cameras` page is served on
#   --log-dir DIR          where run logs go (default: runs/kitchen-arm)
#   --                     everything after this goes to inspect-robot
#
# What the task brings, and what the engines bring. `serve` and `inspect` stage
# shared/tasks/apple_on_plate.py into whichever kitchen an engine compiled: the reference
# rig's work surface and everything on it -- a 20 mm apple, a white plate, and the bowl,
# mug, banana and lemon it keeps as scenery -- at the poses and contact parameters that
# were measured there. The engine still supplies the room and still mounts the arm on its
# counter; the task supplies the geometry, because that is the part a procedurally chosen
# object cannot.
#
# Bringing a table into a kitchen sounds redundant and is not. A camera's framing is a
# property of the surface under it: with the reference slab staged, all four of its
# corners land in the overhead frame at a worst normalised radius of 0.930, which is the
# reference rig's own figure -- against a bare counter running diagonally across the frame
# with a third of it floor. `--no-reference-table` is the other arrangement, and the
# scripted policy is verified both ways.
#
# `--objects`/`--target` still choose what the *engines* put on their worktops, which is
# what `shot` and `view` compare. They are not passed to `serve`/`inspect`: a task that
# stages its own apple does not want a second one the jaw cannot close on.
#
set -euo pipefail

# Job control, so every backgrounded engine becomes its own process group and can be
# killed as one. It has to be: `molmospaces ... &` backgrounds a *function* whose body is
# a subshell, so the python that actually holds the port is two forks below the PID `$!`
# reports. Killing that PID alone reaps the shells and orphans the engine, which then
# sits on its port until the next run fails the port check and blames the port.
set -m

# Resolved without cd; see the note in each engine's env.sh about title-escape capture.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
ROOT="$(dirname "$_self")"
ROOT="$(realpath "$ROOT" 2>/dev/null || echo "${ROOT%/.}")"

MOLMO="$ROOT/molmospaces"
ROBOCASA="$ROOT/robocasa"
CONSOLE="$ROOT/../robot_console"
CONSOLE="$(realpath "$CONSOLE" 2>/dev/null || echo "$CONSOLE")"
# Two console venvs, and which one runs depends on the policy. torch is a multi-gigabyte
# dependency that only the VLA needs, and keeping it out of `.venv` is what makes the
# offline test suite fast and the scripted baseline cheap to run.
CONSOLE_VENV="$CONSOLE/.venv"
CONSOLE_VLA_VENV="$CONSOLE/.venv-vla"

OBJECTS="bowl,apple"
SCENE="ithor:1"
LAYOUT=1
STYLE=1
OUT="$ROOT/shots"
PORTS="9090,9091"
POLICY="so101_waypoint"
ENGINE="molmospaces"
STEPS=400
EPISODES=1
WRIST=0
VIEWER=0
HTTP_PORT=8791
LOG_DIR=""
declare -a STAGE_FLAGS=()
# Set by the commands that stage a task, so the engine wrappers know not to also ask the
# engine for loose objects of its own.
STAGING_TASK=0
declare -a PASSTHRU=()

die() { echo "error: $*" >&2; exit 1; }
say() { printf '\033[1m%s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- arguments

cmd="shot"
case "${1:-}" in
  shot|view|serve|inspect|cameras|help|-h|--help) cmd="$1"; shift || true ;;
esac

while [ $# -gt 0 ]; do
  case "$1" in
    --objects)  OBJECTS="$2"; shift 2 ;;
    --scene)    SCENE="$2";   shift 2 ;;
    --layout)   LAYOUT="$2";  shift 2 ;;
    --style)    STYLE="$2";   shift 2 ;;
    --out)      OUT="$2";     shift 2 ;;
    --ports)    PORTS="$2";   shift 2 ;;
    --policy)   POLICY="$2";  shift 2 ;;
    --engine)   ENGINE="$2";  shift 2 ;;
    --steps)    STEPS="$2";   shift 2 ;;
    --episodes) EPISODES="$2"; shift 2 ;;
    --log-dir)  LOG_DIR="$2"; shift 2 ;;
    --wrist)    WRIST=1;      shift ;;
    --viewer)   VIEWER=1;     shift ;;
    --http-port) HTTP_PORT="$2"; shift 2 ;;
    --no-reference-table) STAGE_FLAGS+=(--no-reference-table); shift ;;
    --no-dressing)        STAGE_FLAGS+=(--no-dressing);        shift ;;
    --reference-lighting) STAGE_FLAGS+=(--reference-lighting); shift ;;
    --extra-lights)       STAGE_FLAGS+=(--extra-lights);       shift ;;
    --)         shift; PASSTHRU=("$@"); break ;;
    *) die "unknown flag '$1' (try: ./kitchen.sh help)" ;;
  esac
done

if [ "$cmd" = "help" ] || [ "$cmd" = "-h" ] || [ "$cmd" = "--help" ]; then
  awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$0"
  exit 0
fi

MOLMO_PORT="${PORTS%%,*}"
ROBOCASA_PORT="${PORTS##*,}"
[ -n "$LOG_DIR" ] || LOG_DIR="$ROOT/runs/kitchen-arm"

case "$ENGINE" in molmospaces|robocasa|both) ;; *) die "--engine: expected molmospaces, robocasa or both" ;; esac

need_engine() {
  [ -x "$1/.venv/bin/python" ] \
    || die "$(basename "$1") is not set up yet - run: cd $1 && ./run.sh setup"
}
# `shot` and `view` are comparisons and need both engines by definition. `serve` and
# `inspect` take --engine, so they need only the ones they were asked for -- setting up
# the second engine is a large download, and requiring it to run the first is a barrier
# with nothing behind it. `cameras` starts no engine at all.
case "$cmd" in
  cameras) ;;
  serve|inspect)
    [ "$ENGINE" = "robocasa" ]    || need_engine "$MOLMO"
    [ "$ENGINE" = "molmospaces" ] || need_engine "$ROBOCASA"
    ;;
  *) need_engine "$MOLMO"; need_engine "$ROBOCASA" ;;
esac

# ---------------------------------------------------------------- engines
#
# Each engine runs in its own subshell. `env.sh` exports VENV_DIR, PYTHONPATH and
# MUJOCO_GL, and the two engines disagree on all three -- sourcing both into one shell
# would put robosuite on MolmoSpaces' path and the wrong interpreter on both.

molmospaces() {
  (
    # shellcheck source=/dev/null
    source "$MOLMO/env.sh"
    local xml
    # Scene MJCFs reference their meshes through the assets/ symlink tree, so the scene
    # has to be named by its assets/ path and not by its realpath. resolve_scene.py is
    # what gets that right -- and it downloads the house if this is a first run.
    xml="$("$MOLMO/.venv/bin/python" "$MOLMO/tools/resolve_scene.py" \
            "${SCENE%%:*}" "${SCENE##*:}")" || die "could not resolve scene $SCENE"
    # exec, so the PID this subshell reports to `$!` IS the engine. Without it the
    # subshell forks python as a child, `kill $!` reaps only the subshell, and the engine
    # is orphaned still holding its port -- which the next run then fails on, blaming the
    # port rather than the leak.
    exec "$1" "$MOLMO/tools/spawn_robot.py" so101 --scene "$xml" --target "$OBJECTS" "${@:2}"
  )
}

robocasa() {
  (
    # shellcheck source=/dev/null
    source "$ROBOCASA/env.sh"
    # `--objects` only for the comparison commands. `serve`/`inspect` stage a task,
    # which brings its own objects at measured positions -- and RoboCasa's sampler would
    # add a second apple the jaw cannot close on plus a bowl inside the plate's
    # footprint. spawn_robot.py refuses the combination outright; this is what keeps it
    # from ever being asked for.
    local objects=()
    [ "$STAGING_TASK" -eq 1 ] || objects=(--objects "$OBJECTS")
    exec "$1" "$ROBOCASA/tools/spawn_robot.py" so101 --layout "$LAYOUT" --style "$STYLE" \
      ${objects[@]+"${objects[@]}"} "${@:2}"
  )
}

# The MuJoCo passive viewer must own the main thread on macOS, which is what mjpython
# provides; anything windowless runs under plain python. Same rule as both run.sh files.
py()     { echo "$1/.venv/bin/python"; }
# --viewer needs mjpython on macOS (the passive viewer must own the Cocoa main thread)
# and drops --headless. The offscreen camera renderers and the on-screen window then
# coexist in one process, which was not obviously safe beforehand -- under mjpython the
# script runs off the main thread and each renderer opens a hidden GLFW window -- and was
# verified before this flag was added. If it ever stops working, both env.sh files name
# MUJOCO_GL=cgl as the escape hatch.
engine_python() {
  local root; root="$(engine_root "$1")"
  if [ "$VIEWER" -eq 1 ]; then viewer "$root"; else py "$root"; fi
}
headless_arg() { [ "$VIEWER" -eq 1 ] || echo "--headless"; }
viewer() {
  if [ "$(uname -s)" = "Darwin" ]; then echo "$1/.venv/bin/mjpython"; else echo "$1/.venv/bin/python"; fi
}

# 9090 is the rosbridge default and what a real bringup for this arm uses, so it is the
# right default here -- but it is also a popular port, and a sibling checkout running its
# own simulator is the likeliest thing holding it. Naming the holder turns a puzzling
# failure into an obvious one.
# The engines this invocation covers, as `name:port`, honouring --engine.
engine_list() {
  case "$ENGINE" in
    molmospaces) echo "molmospaces:$MOLMO_PORT" ;;
    robocasa)    echo "robocasa:$ROBOCASA_PORT" ;;
    both)        echo "molmospaces:$MOLMO_PORT robocasa:$ROBOCASA_PORT" ;;
  esac
}
engine_root() { [ "$1" = molmospaces ] && echo "$MOLMO" || echo "$ROBOCASA"; }

port_free() {
  nc -z 127.0.0.1 "$1" 2>/dev/null || return 0
  local holder
  holder="$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -Fc 2>/dev/null | sed -n 's/^c//p' | sort -u | paste -sd, -)"
  die "port $1 is already in use${holder:+ (by: $holder)} - pick others with --ports A,B"
}

# Kill a backgrounded engine and everything it forked, then wait for the port to actually
# come free -- the next run's port check is otherwise the first thing that notices.
stop_engine() {
  local pid="$1" port="$2" i
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  for i in $(seq 1 20); do
    nc -z 127.0.0.1 "$port" 2>/dev/null || return 0
    sleep 0.5
  done
  echo "warning: port $port still held after stopping the engine" >&2
}

# Wait for the topics, not for the port. A listening socket only means the websocket
# server bound; it says nothing about whether the scene compiled, the cameras resolved,
# or the task staged. Every one of those failures otherwise arrives minutes later as a
# client-side reset timeout with no clue in it.
wait_for_topics() {
  local url="$1" pidfile="$2" label="$3" i
  for i in $(seq 1 180); do
    if [ -n "$pidfile" ] && ! kill -0 "$pidfile" 2>/dev/null; then
      echo >&2; die "$label died during startup - see the log above"
    fi
    if "$CONSOLE/.venv/bin/python" -m robot_console.arm.preflight \
         --url "$url" --no-reset --allow-out-of-reach >/dev/null 2>&1; then
      echo " ready (${i}s)" >&2
      return 0
    fi
    [ $((i % 5)) -eq 0 ] && printf '.' >&2
    sleep 1
  done
  echo >&2; die "$label never published its topics on $url"
}

# ---------------------------------------------------------------- commands

case "$cmd" in
  shot)
    mkdir -p "$OUT"
    echo ">> molmospaces: so101 in $SCENE around $OBJECTS"
    molmospaces "$(py "$MOLMO")" --render "$OUT/molmospaces_so101.png" \
      --width 1600 --height 1000 --distance 1.1 --elevation -22
    echo ">> robocasa: so101 in kitchen layout $LAYOUT style $STYLE around $OBJECTS"
    robocasa "$(py "$ROBOCASA")" --render "$OUT/robocasa_so101.png" \
      --width 1600 --height 1000 --distance 1.1 --elevation -22
    echo
    echo "screenshots:"
    echo "  $OUT/molmospaces_so101.png"
    echo "  $OUT/robocasa_so101.png"
    ;;

  view)
    # Two windows, two processes. Backgrounding the first is safe precisely because they
    # are separate processes: each mjpython owns the main thread of its own.
    echo ">> molmospaces viewer (close the window to quit)"
    molmospaces "$(viewer "$MOLMO")" &
    molmo_pid=$!
    echo ">> robocasa viewer (close the window to quit)"
    robocasa "$(viewer "$ROBOCASA")" &
    robocasa_pid=$!
    trap 'kill "$molmo_pid" "$robocasa_pid" 2>/dev/null || true' INT TERM
    wait
    ;;

  serve)
    STAGING_TASK=1
    wrist_arg=(); [ "$WRIST" -eq 1 ] && wrist_arg=(--wrist-camera)
    pids=()
    # EXIT as well as INT/TERM: without it a `die` anywhere below leaves an engine
    # holding its port, and the next run fails the port check for no visible reason.
    trap 'for i in "${!pids[@]}"; do stop_engine "${pids[$i]}" "${serve_ports[$i]}"; done' INT TERM EXIT
    serve_ports=()
    for spec in $(engine_list); do
      name="${spec%%:*}"; port="${spec##*:}"
      # Checked up front, because the failure otherwise arrives as a websockets
      # traceback from an engine that has already spent a minute compiling a kitchen.
      port_free "$port"
      echo ">> $name so101 on ws://127.0.0.1:$port$([ "$VIEWER" -eq 1 ] && echo ' (with a window)')"
      "$name" "$(engine_python "$name")" $(headless_arg) --ros-port "$port" \
        --task apple_on_plate --control-hz 10 "${wrist_arg[@]}" \
        ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
      pids+=("$!"); serve_ports+=("$port")
    done
    echo
    echo "drive any of them with the same client, from robot_console/:"
    echo "  inspect-robot run --task apple_on_plate --policy so101_waypoint \\"
    echo "      --embodiment so101_ros -E url=ws://127.0.0.1:${serve_ports[0]} \\"
    echo "      -E fresh_obs_timeout_s=2.0 -T max_steps=$STEPS --max-action-delta 0.65"
    echo "  ./kitchen.sh cameras --ports ${serve_ports[0]},$ROBOCASA_PORT"
    wait
    ;;

  cameras)
    page="$ROOT/live_cameras.html"
    [ -f "$page" ] || die "missing $page"
    # Follows --engine, so `cameras --engine robocasa` does not silently point at the
    # other engine's port.
    spec="$(engine_list)"; spec="${spec%% *}"
    port="${spec##*:}"; url="ws://127.0.0.1:$port"
    nc -z 127.0.0.1 "$port" 2>/dev/null \
      || die "nothing is serving on $url - start one with ./kitchen.sh serve"

    # Served over HTTP rather than opened from file://, because browsers refuse a ws://
    # connection from a file:// origin and the page then sits there discovering nothing.
    port_free "$HTTP_PORT"
    python3 -m http.server "$HTTP_PORT" --directory "$ROOT" --bind 127.0.0.1 \
      >/dev/null 2>&1 &
    http_pid=$!
    trap 'kill "$http_pid" 2>/dev/null || true' INT TERM EXIT
    page_url="http://127.0.0.1:$HTTP_PORT/live_cameras.html?url=$url"
    say "camera page: $page_url"
    echo "  cameras are discovered from the wire, so --wrist shows up without a reload"
    echo "  the sliders drive the arm once you tick 'Enable control'"
    echo "  Ctrl-C stops serving the page; the simulator keeps running"
    command -v open >/dev/null && open "$page_url"
    wait "$http_pid"
    ;;

  inspect)
    # The VLA policy needs torch, which lives only in .venv-vla; everything else runs in
    # the light venv. Choosing here rather than making the caller remember it.
    case "$POLICY" in
      molmoact2) RUN_VENV="$CONSOLE_VLA_VENV"; extra="vla" ;;
      *)         RUN_VENV="$CONSOLE_VENV";     extra="dev,arm" ;;
    esac
    [ -x "$RUN_VENV/bin/inspect-robot" ] || die \
      "inspect-robot is not installed in $(basename "$RUN_VENV") - run:
    cd $CONSOLE && uv venv --python 3.12 $(basename "$RUN_VENV") \
        && VIRTUAL_ENV=$(basename "$RUN_VENV") uv pip install -e '.[$extra]'"

    STAGING_TASK=1
    overall=0
    for spec in $(engine_list); do
      name="${spec%%:*}"; port="${spec##*:}"; url="ws://127.0.0.1:$port"
      port_free "$port"
      wrist_arg=(); [ "$WRIST" -eq 1 ] && wrist_arg=(--wrist-camera)

      say "== $name: staging apple_on_plate on $url"
      "$name" "$(engine_python "$name")" $(headless_arg) \
        --ros-port "$port" --task apple_on_plate --control-hz 10 "${wrist_arg[@]}" \
        ${STAGE_FLAGS[@]+"${STAGE_FLAGS[@]}"} &
      sim_pid=$!
      trap 'stop_engine "$sim_pid" "$port"' INT TERM EXIT

      printf 'waiting for topics' >&2
      wait_for_topics "$url" "$sim_pid" "$name"

      passes=0
      for episode in $(seq 1 "$EPISODES"); do
        run_dir="$LOG_DIR/$name/$(date +%Y%m%d-%H%M%S)-$episode"
        # The preflight is separate from the episode's own reset for a reason: /reset
        # says the world was restored, and this checks that it *was* -- the apple
        # measured back at spawn and the scripted plan still solving from there. A
        # drifted apple otherwise scores zero looking exactly like a policy failure.
        if ! "$CONSOLE/.venv/bin/python" -m robot_console.arm.preflight --url "$url" \
               --json "$run_dir/scene_reset.json"; then
          case $? in
            2) die "no simulator answering on $url" ;;
            3) die "the world did not reset - restart the engine" ;;
            4) die "the apple is where the arm cannot complete the plan" ;;
            *) die "preflight failed" ;;
          esac
        fi

        # fresh_obs_timeout_s: how long a step waits for an observation newer than the
        # command it just sent. The default is 2/control_hz = 0.2 s, which assumes the
        # simulator publishes faster than the control rate -- and this one does not: it
        # renders its cameras inside the physics loop and manages 8-10 Hz, and a VLA adds
        # seconds of inference on top. Measured: an episode died mid-run with
        # "EmbodimentFault: no post-publish joint state within fresh_obs_timeout_s=0.2s".
        # Two seconds is generous, and it is still a freshness guarantee -- a stale
        # observation is refused, it is just given time to arrive.
        #
        # --max-action-delta 0.65 raises the framework's own per-step limiter above the
        # policy's largest intended single-step change (the jaw closing 1.0 -> 0.40).
        # The default is derived from the action space and lands near 0.03, which halves
        # the arm's 0.06 rad step: measured, `home` and `pre_grasp` then burn their whole
        # 40-step budget without arriving and the episode runs out of steps in transit.
        # The policy is already the rate limiter, and it is the one holding the measured
        # constants; this leaves the bounds clamp in place and gets the limiter out of
        # its way.
        set +e
        "$RUN_VENV/bin/inspect-robot" run \
          --task apple_on_plate --policy "$POLICY" --embodiment so101_ros \
          -E "url=$url" -E "fresh_obs_timeout_s=2.0" -T "max_steps=$STEPS" \
          --max-action-delta 0.65 --grader none --no-prompt \
          --log-dir "$run_dir" ${PASSTHRU[@]+"${PASSTHRU[@]}"} > "$run_dir/episode.log" 2>&1
        status=$?
        set -e
        # The verdict is decided in Python, not by string-matching in the shell: these
        # scores are floats on the wire (`1.0`, not `1`), and `[ "$x" = "1" ]` quietly
        # calls a passing episode a failure. Measured the hard way, on an episode that
        # placed the apple 13 mm from the plate centre and was reported FAIL.
        verdict="$("$CONSOLE/.venv/bin/python" - "$run_dir" <<'PY'
import glob, json, sys
files = [f for f in glob.glob(sys.argv[1] + "/*.json") if "scene_reset" not in f]
if not files:
    print("FAIL FAIL nan")
    raise SystemExit
log = json.load(open(files[0]))
sample = log["samples"][0]
# An episode that *errored* and one that ran and scored zero are different failures, and
# reporting both as "apple nan m from plate centre" hides the first behind the second.
if log.get("status") != "success" or sample.get("error"):
    reason = str(sample.get("error") or log.get("error") or "errored").splitlines()[0]
    print(f"ERROR ERROR {reason[:90]}")
    raise SystemExit
scores = sample["epochs"][0]
scored = "PASS" if float(scores.get("apple_on_plate", 0)) >= 1.0 else "FAIL"
sim = "PASS" if float(scores.get("sim_task_success", 0)) >= 1.0 else "FAIL"
print(f"{scored} {sim} {scores.get('apple_plate_distance', float('nan')):.4f} m from plate centre")
PY
)"
        read -r scored sim_said detail <<<"$verdict"
        if [ "$scored" = "PASS" ] && [ "$status" -eq 0 ]; then
          passes=$((passes + 1))
          printf '  episode %s/%s: \033[32mPASS\033[0m  apple %s' \
            "$episode" "$EPISODES" "$detail"
        elif [ "$scored" = "ERROR" ]; then
          printf '  episode %s/%s: \033[31mERROR\033[0m %s' "$episode" "$EPISODES" "$detail"
        else
          printf '  episode %s/%s: \033[31mFAIL\033[0m  apple %s (gate 0.080)' \
            "$episode" "$EPISODES" "$detail"
        fi
        # The simulator's own verdict and the offline scorer are computed independently
        # and are reported separately on purpose. They can legitimately disagree at the
        # margin -- the episode terminates the instant its own hold passes 1.0 s, and the
        # simulator's hold starts a beat later -- and a disagreement is information, not
        # something to paper over by having one read the other.
        [ "$sim_said" = "$scored" ] \
          || printf '  \033[33m[disagreement: /task_success said %s]\033[0m' "$sim_said"
        echo "   log: $run_dir"
      done

      say "== $name: $passes/$EPISODES passed"
      [ "$passes" -gt 0 ] || overall=1
      stop_engine "$sim_pid" "$port"
      trap - INT TERM EXIT
    done
    exit "$overall"
    ;;
esac
