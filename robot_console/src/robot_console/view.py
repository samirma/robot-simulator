"""``view.sh``: serve the camera and bounded-control page, which speaks rosbridge itself.

    view.sh [--url ws://host:port] [--robot <id>] [--namespace <name>] [--no-open] [--port N]

The page (``web/view.html`` with its inline styles + ``web/console.js``; no external assets, so
it works on robot networks without internet access) is static: it is handed the websocket URL,
the optional robot/namespace selection and the packaged profiles (``/profiles.json``: the
same profiles teleop and fleet use, with the controls, measured reads and 3D ``model`` the
page needs) and does discovery, validation, cameras, the model and controls in the browser. The
server binds to 127.0.0.1 and serves only those files.
"""

from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
import webbrowser
from importlib import resources
from typing import List, Optional

from robot_console.profiles import SUPPORTED_IDS, ProfileError, all_profiles_json, check_namespace_allowed, select_id
from robot_console.rosbridge import DEFAULT_URL, check_url

STATIC = {"/": ("view.html", "text/html; charset=utf-8"),
          "/view.html": ("view.html", "text/html; charset=utf-8"),
          "/console.js": ("console.js", "application/javascript; charset=utf-8")}


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="view.sh",
        description="Browser page with the selected robot's cameras and the bounded controls its "
                    "profile lists. Changes are sent to the robot immediately and nothing is "
                    "stopped automatically. No base-drive or walking control.",
        epilog=f"Robot ids: {', '.join(SUPPORTED_IDS)}")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"rosbridge websocket (default {DEFAULT_URL})")
    ap.add_argument("--robot", metavar="ID", help="robot profile (default: identify from the wire)")
    ap.add_argument("--namespace", metavar="NAME",
                    help="only for profiles whose hardware interface documents namespaces")
    ap.add_argument("--port", type=int, default=0, help="HTTP port on 127.0.0.1 (default: any free)")
    ap.add_argument("--no-open", action="store_true", help="print the page address, do not open it")
    return ap


def make_server(url: str, robot: Optional[str], namespace: Optional[str], port: int = 0
                ) -> http.server.ThreadingHTTPServer:
    config = json.dumps({"url": url, "robot": robot, "namespace": namespace}).encode()
    profiles = json.dumps(all_profiles_json()).encode()
    web = resources.files("robot_console").joinpath("web")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/config.json":
                body, ctype = config, "application/json"
            elif path == "/profiles.json":
                body, ctype = profiles, "application/json"
            elif path in STATIC:
                name, ctype = STATIC[path]
                body = web.joinpath(name).read_bytes()
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # quiet
            pass

    return http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        check_url(args.url)
        p = select_id(args.robot)
        if p is not None:
            check_namespace_allowed(p, args.namespace)
    except (ProfileError, ValueError) as exc:
        print(f"view: refused: {exc}", file=sys.stderr)
        return 2
    srv = make_server(args.url, args.robot, args.namespace, args.port)
    page = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"view: serving {page} for {args.url} (Ctrl-C to stop)", flush=True)
    if not args.no_open:
        threading.Timer(0.3, lambda: webbrowser.open(page)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
