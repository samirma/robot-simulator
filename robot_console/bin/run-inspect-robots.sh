#!/usr/bin/env bash
# An LLM agent (Inspect Robots) performs a task on the SO-101 in each simulator.
#
#   ./bin/run-inspect-robots.sh                          default task, both simulators
#   ./bin/run-inspect-robots.sh "wave at the camera"     custom task text
#   ./bin/run-inspect-robots.sh --ports 8000             one simulator only
#   ./bin/run-inspect-robots.sh --no-record              skip the feed.mp4 recordings
#   ./bin/run-inspect-robots.sh --help                   this text
#
# Default task: "Pick up the apple and place it inside the bowl." -- matching what
# `simulator/kitchen_arm.sh serve` stages next to the arm in both engines
# (MolmoSpaces on port 8000, RoboCasa on 8001). Start that first. Each simulator's
# episode is recorded to runs/inspect-<port>/feed.mp4 unless --no-record.
#
# The policy is inspect-robots-agent's LLMAgentPolicy: the LLM sees the camera and
# drives the arm through tool calls, with the unmodified inspect-robots-so101
# embodiment talking to the simulator through SimulatorSO101 (robot-console-inspect-so101).
#
# Backend: Qwen3.8-27B-4bit served by oMLX on port $OMLX_PORT (started here if not
# already up). Measured against the same vision + tool-call probe, oMLX answers in
# 8-9 s warm vs llama.cpp's 10.7 s (Q4_K_M + Q8 mmproj), and its prompt caching pays
# again on every call of an agent loop that resends the transcript. To use llama.cpp
# instead:
#   /Users/U124317/llm/llama.cpp/build/bin/llama-server \
#       -m <Qwen3.8-27B-Q4_K_M.gguf> --mmproj <mmproj-Qwen3.8-27B-Q8_0.gguf> \
#       --port 8080 -ngl 99 -c 16384 --jinja
#   ./bin/run-inspect-robots.sh --model whatever
# ChatGPT gpt-5.6-luna (xhigh) was evaluated and is NOT usable here: it exists only
# behind opencode's ChatGPT OAuth, and opencode serves no OpenAI-compatible endpoint
# for other clients to call -- there is nothing to point --base-url at.
set -euo pipefail

# Resolved without cd; see the note in teleop.sh about terminal-title escapes.
_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
CONSOLE_ROOT="$(realpath "$CONSOLE_ROOT" 2>/dev/null || echo "${CONSOLE_ROOT%/.}")"

VENV_DIR="${ROBOT_CONSOLE_VENV:-$CONSOLE_ROOT/.venv}"
PY="$VENV_DIR/bin/python"
# Its own stamp: the [inspect] extra rides in .venv (it is small, unlike torch), but
# a teleop bootstrap must not be forced to reinstall it and vice versa.
STAMP="$VENV_DIR/.inspect-stamp"

OMLX_PORT="${OMLX_PORT:-8080}"
OMLX_MODEL_DIR="$HOME/.omlx/models"

die() { echo "error: $*" >&2; exit 1; }

usage() { awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "$_self"; }

bootstrap() {
  command -v uv >/dev/null || die "uv not found; install it with
    curl -LsSf https://astral.sh/uv/install.sh | sh"
  [ -x "$PY" ] || { echo ">> creating venv ($VENV_DIR)"; uv venv --python 3.12 "$VENV_DIR" || uv venv "$VENV_DIR"; }
  echo ">> installing robot_console[inspect]"
  VIRTUAL_ENV="$VENV_DIR" uv pip install -e "$CONSOLE_ROOT[inspect]"
  touch "$STAMP"
}

# ---------------------------------------------------------------- arguments

TASK="Pick up the apple and place it inside the bowl."
PORTS="8000,8001"
MODEL="mlx-community--Qwen3.8-27B-4bit"
BASE_URL="http://127.0.0.1:$OMLX_PORT/v1"
RECORD=1
# One move_joints call precomputes a linear ramp of ceil(max|delta|/step_limit)
# embodiment steps, with step_limit = min(max_speed_frac/control_hz, 0.05) x range,
# and the LLM is not consulted again until the ramp finishes -- while Task.max_steps
# counts those embodiment steps. The first recorded runs shipped the inspect_so101
# defaults (speed 0.05, steps 30): a single 62-degree move planned 113 steps, the
# trial ended 30 steps in, and the whole episode moved ~16 degrees. Speed 0.5 makes a
# move ~12 steps, so every call returns to the LLM quickly (closed loop), and 400
# steps is 20 s of actual motion. LLM calls are the scarce resource at 30-60 s each;
# 20 covers a reach-grasp-lift-place with slack for alignment corrections.
MAX_STEPS=400
MAX_LLM_CALLS=20
MAX_SPEED_FRAC=0.5
LEARNINGS="$BIN_DIR/so101_learnings.md"
declare -a EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --reinstall) bootstrap; exit 0 ;;
    --task) TASK="$2"; shift 2 ;;
    --ports) PORTS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --no-record) RECORD=0; shift ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-llm-calls) MAX_LLM_CALLS="$2"; shift 2 ;;
    --max-speed-frac) MAX_SPEED_FRAC="$2"; shift 2 ;;
    --learnings) LEARNINGS="$2"; shift 2 ;;
    --*) EXTRA+=("$1" "$2"); shift 2 ;;
    *) TASK="$*"; break ;;
  esac
done

if [ ! -x "$PY" ] || [ ! -f "$STAMP" ] || [ "$CONSOLE_ROOT/pyproject.toml" -nt "$STAMP" ]; then
  bootstrap
fi
"$PY" -c "import inspect_robots" 2>/dev/null || bootstrap

# ---------------------------------------------------------------- LLM backend

# oMLX discovers models from subdirectories of ~/.omlx/models; the Qwen weights live
# in the Hugging Face cache, so a symlink is the whole installation.
if [ ! -e "$OMLX_MODEL_DIR/$MODEL" ] && [ "$MODEL" = "mlx-community--Qwen3.8-27B-4bit" ]; then
  snap="$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/ 2>/dev/null | head -1)"
  [ -n "$snap" ] && { mkdir -p "$OMLX_MODEL_DIR"; ln -sfn "${snap%/}" "$OMLX_MODEL_DIR/$MODEL"; }
fi

if ! curl -s --max-time 2 "$BASE_URL/models" -H "Authorization: Bearer 1234" >/dev/null 2>&1; then
  case "$BASE_URL" in
    *127.0.0.1:$OMLX_PORT*|*localhost:$OMLX_PORT*)
      command -v omlx >/dev/null || die "omlx not found and nothing serving at $BASE_URL"
      echo ">> starting omlx on port $OMLX_PORT"
      nohup omlx serve --port "$OMLX_PORT" > /tmp/omlx-inspect.log 2>&1 &
      for _ in $(seq 1 30); do
        curl -s --max-time 2 "$BASE_URL/models" -H "Authorization: Bearer 1234" >/dev/null 2>&1 && break
        sleep 2
      done
      curl -s --max-time 2 "$BASE_URL/models" -H "Authorization: Bearer 1234" >/dev/null 2>&1 \
        || die "omlx did not come up on $BASE_URL (see /tmp/omlx-inspect.log)"
      ;;
    *) die "nothing serving at $BASE_URL; start your backend first" ;;
  esac
fi

# oMLX's configured API key; LLMAgentPolicy reads the key from an env var by name.
export OMLX_API_KEY="${OMLX_API_KEY:-1234}"

# ---------------------------------------------------------------- run

status=0
IFS=',' read -ra port_list <<< "$PORTS"
for port in "${port_list[@]}"; do
  echo
  echo "=== simulator on port $port ==="
  args=(--port "$port" --instruction "$TASK"
        --model "$MODEL" --base-url "$BASE_URL" --api-key-env OMLX_API_KEY
        --max-steps "$MAX_STEPS" --max-llm-calls "$MAX_LLM_CALLS"
        --max-speed-frac "$MAX_SPEED_FRAC"
        --log-dir "runs/inspect-$port")
  if [ -f "$LEARNINGS" ]; then
    # The joint-0 sign relative to the image differs between the two stagings (the
    # mounts face different ways and the cameras sit on different sides), and an agent
    # that guesses it wrong once tends to lock the wrong sign in and ride a joint
    # limit to a confident give_up. These lines are rig facts verified from recorded
    # episodes -- exactly what a learnings file is for.
    run_learnings="$(mktemp -t so101-learnings)"
    cp "$LEARNINGS" "$run_learnings"
    case "$port" in
      8000) printf -- '- Verified on this staging: the apple sits image-right and is reached with joint 0 around +35 to +55.\n' >> "$run_learnings" ;;
      8001) printf -- '- Verified on this staging (from scene geometry, not guesswork): the apple is at joint 0 = +50 and the bowl at joint 0 = -50, both 25 cm from the base. If your image reading disagrees, trust these numbers.\n' >> "$run_learnings" ;;
    esac
    args+=(--prior-learnings "$run_learnings")
  fi
  [ "$RECORD" = 1 ] && args+=(--record "runs/inspect-$port")
  "$PY" -m robot_console.inspect_so101 "${args[@]}" "${EXTRA[@]+"${EXTRA[@]}"}" || status=1
done
exit $status
