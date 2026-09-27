"""The rosbridge wire an entry point talks to: `--url ws://<host>:<port>`.

Every console entry point takes the same flag with the same default (console spec §1), so
the parsing lives here once. It is stdlib only, which lets a module like `fleet.py` or
the arm's preflight share it without importing anything else of the teleop's.
"""

from __future__ import annotations

import argparse
from typing import Tuple
from urllib.parse import urlsplit

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9090
DEFAULT_URL = f"ws://{DEFAULT_HOST}:{DEFAULT_PORT}"

SCHEMES = ("ws", "wss")


def parse_url(url: str) -> Tuple[str, int]:
    """`(host, port)` from `ws://host:port`. Raises `ValueError` for anything else.

    The port may be omitted (it defaults to 9090); a path, query or credentials may not,
    because rosbridge serves none and silently ignoring one would hide a typo.
    """
    text = (url or "").strip()
    parts = urlsplit(text)
    if parts.scheme not in SCHEMES:
        raise ValueError(f"{url!r} is not a rosbridge URL; expected ws://<host>:<port>")
    if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username:
        raise ValueError(f"{url!r}: a rosbridge URL is ws://<host>:<port> and nothing else")
    host = parts.hostname
    if not host:
        raise ValueError(f"{url!r} names no host")
    try:
        port = parts.port
    except ValueError:
        raise ValueError(f"{url!r} has an invalid port") from None
    return host, int(port if port is not None else DEFAULT_PORT)


def normalise_url(url: str) -> str:
    """`url` in canonical `ws://host:port` form (IPv6 hosts bracketed)."""
    host, port = parse_url(url)
    scheme = urlsplit(url.strip()).scheme
    shown = f"[{host}]" if ":" in host else host
    return f"{scheme}://{shown}:{port}"


def url_arg(value: str) -> str:
    """An argparse `type=` for `--url`: validates and canonicalises."""
    try:
        return normalise_url(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_url_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url", type=url_arg, default=DEFAULT_URL, metavar="ws://HOST:PORT",
        help=f"the rosbridge to talk to (default {DEFAULT_URL})",
    )
