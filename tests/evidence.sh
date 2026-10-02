#!/usr/bin/env bash
# Workspace evidence (workspace spec §3): runs tests/evidence.py in a throwaway uv
# environment that depends on neither project.
#
#   tests/evidence.sh                                    # every robot on both engines
#   tests/evidence.sh --engine molmospaces --robot myagv # a subset (flags repeat)
#   tests/evidence.sh --index-only                       # rebuild evidences/index.{md,json}
#   tests/evidence.sh --help
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --no-project --quiet --with websocket-client --with numpy --with pillow --with pyyaml \
  python -u "$HERE/evidence.py" "$@"
