#!/usr/bin/env bash
# The camera and control page for whatever is on a rosbridge wire.
#
#   ./bin/view.sh                          the fleet on ws://127.0.0.1:9090
#   ./bin/view.sh --url ws://10.0.0.7:9090 ...on another wire
#   ./bin/view.sh --http-port 8791         serve the page on a fixed port (default: any free one)
#   ./bin/view.sh --no-open                print the page's address instead of opening it
#
# One static page, `live_cameras.html`, which speaks rosbridge from the browser. It is told
# the URL and nothing else: members and cameras come from `/rosapi`, typed. A robot's panel
# sends nothing until its "Enable control" box is ticked, and offers only bounded commands
# on the robot's official interface -- never a myAGV drive or an AiNex walk; drive those
# with bin/teleop.sh, which can guarantee the stop a browser tab cannot.
#
# Needs only a python3 on PATH (the standard library's HTTP server); no venv. The server
# binds to 127.0.0.1 and serves that one page, and stops with Ctrl-C.
set -euo pipefail

_self="${BASH_SOURCE[0]}"
case "$_self" in /*) ;; *) _self="$PWD/$_self" ;; esac
BIN_DIR="$(dirname "$_self")"
CONSOLE_ROOT="$(dirname "$BIN_DIR")"
PAGE="$CONSOLE_ROOT/live_cameras.html"

URL="ws://127.0.0.1:9090"
HTTP_PORT=0
OPEN=1

die() { echo "error: $*" >&2; exit 2; }
usage() { sed -n '2,15p' "$_self" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --url) [ $# -ge 2 ] || die "--url needs a value"; URL="$2"; shift 2 ;;
    --url=*) URL="${1#--url=}"; shift ;;
    --http-port) [ $# -ge 2 ] || die "--http-port needs a value"; HTTP_PORT="$2"; shift 2 ;;
    --http-port=*) HTTP_PORT="${1#--http-port=}"; shift ;;
    --no-open) OPEN=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

case "$URL" in
  ws://?*|wss://?*) ;;
  *) die "--url must be ws://<host>:<port> or wss://..., got '$URL'" ;;
esac
case "$HTTP_PORT" in ''|*[!0-9]*) die "--http-port must be a number" ;; esac
[ -f "$PAGE" ] || die "missing $PAGE"
command -v python3 >/dev/null || die "python3 not found"

exec python3 - "$PAGE" "$URL" "$HTTP_PORT" "$OPEN" <<'PY'
import http.server
import subprocess
import sys
import urllib.parse
import webbrowser

page_path, url, port, want_open = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
with open(page_path, "rb") as f:
    PAGE = f.read()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path not in ("/", "/live_cameras.html"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(PAGE)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(PAGE)

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
address = f"http://127.0.0.1:{server.server_address[1]}/?" + urllib.parse.urlencode({"url": url})
print(f"view: {address}  (Ctrl-C to stop)", flush=True)
if want_open:
    opener = {"darwin": ["open"], "linux": ["xdg-open"]}.get(sys.platform)
    try:
        subprocess.Popen(opener + [address]) if opener else webbrowser.open(address)
    except OSError:
        webbrowser.open(address)
try:
    server.serve_forever()
except KeyboardInterrupt:
    pass
PY
