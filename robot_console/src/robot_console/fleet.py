"""Non-interactive wire check: ``python -m robot_console.fleet [--url ws://…] [--expect <id>]``.

Discovers the robots on a rosbridge wire through typed signatures, validates each against
its packaged profile (every required endpoint with its type, and nothing else under the
profile's names but optional rows and ROS infrastructure), and reports each profile camera
as live, stale or missing. It publishes nothing: it only calls rosapi and subscribes.

Exit status: 0 pass; 1 validation failed, an expected camera is not live, ambiguous
candidates, or no supported robot; 2 bad arguments (unknown/assembly id); 3 wire unreachable.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional

from robot_console import camera as cam
from robot_console.discovery import (
    DiscoveryError, SelectionError, Target, describe_candidates, discover, fetch_graph, select_target,
)
from robot_console.profiles import SUPPORTED_IDS, ProfileError, select_id
from robot_console.rosbridge import DEFAULT_URL, Rosbridge, TransportError, check_url

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_UNREACHABLE = 0, 1, 2, 3


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m robot_console.fleet",
        description="Validate the robots on a rosbridge wire against the console's profiles "
                    "and check their cameras. Publishes nothing.",
        epilog=f"Supported robot ids: {', '.join(SUPPORTED_IDS)}")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"rosbridge websocket (default {DEFAULT_URL})")
    ap.add_argument("--expect", metavar="ID", help="require exactly this robot's whole typed interface")
    ap.add_argument("--timeout", type=float, default=5.0, help="rosapi call timeout, s (default 5)")
    return ap


def check_cameras(rb: Rosbridge, targets: List[Target], graph, out) -> bool:
    """Sample every profile camera; True when every non-optional one is live."""
    specs = []
    for t in targets:
        specs += cam.target_specs(t)
    if not specs:
        for t in targets:
            print(f"  cameras: none in profile '{t.profile.id}'", file=out)
        return True
    cams = cam.CameraSet(rb, specs, present=graph.topics)
    window = max(s.stale_after_s for s in specs) * 2 + 0.5
    end = time.monotonic() + window
    while time.monotonic() < end:
        cams.poll()
        if all(s.missing or s.frames >= 3 for s in cams.streams.values()):
            break
        time.sleep(0.05)
    cams.poll()
    ok = True
    now = time.monotonic()
    for s in cams.streams.values():
        st = s.state(now)
        if st == cam.LIVE and s.frames < 2:
            st = cam.STALE          # a single frame is not evidence of a live stream
        label = {cam.LIVE: "live", cam.MISSING: "missing"}.get(st, "stale" if st in (cam.STALE, cam.WAITING) else st)
        detail = s.status_text(now) if st != cam.LIVE else f"live ({s.frames} frames in {window:.1f} s window)"
        flag = " (optional)" if s.spec.optional else ""
        print(f"  camera {s.spec.topic}{flag}: {label.upper()} - {detail}", file=out)
        if st != cam.LIVE and not s.spec.optional:
            ok = False
    cams.close()
    return ok


def report_target(t: Target, out) -> None:
    v = t.validation
    req = len(t.profile.required())
    verdict = "PASS" if v.ok else "FAIL"
    print(f"robot {t.label}: typed validation {verdict} ({req} required endpoints"
          f"{', ' + str(len(v.optional_present)) + ' optional present' if v.optional_present else ''})",
          file=out)
    for p in v.problems():
        print(f"  - {p}", file=out)
    for k, n, why in v.unverified:
        print(f"  note: {k} {n} present; type not verifiable ({why})", file=out)


def run(url: str, expect: Optional[str], timeout: float = 5.0, out=sys.stdout) -> int:
    try:
        check_url(url)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        profile = select_id(expect) if expect else None
    except ProfileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    rb = Rosbridge(url)
    try:
        rb.connect(timeout)
    except TransportError as exc:
        print(f"FAIL: wire unreachable: {exc}", file=out)
        return EXIT_UNREACHABLE
    try:
        try:
            g = fetch_graph(rb, timeout)
        except DiscoveryError as exc:
            print(f"FAIL: cannot read the wire: {exc}", file=out)
            return EXIT_UNREACHABLE if not rb.connected else EXIT_FAIL
        print(f"wire {url}: {(g.dialect or 'unknown dialect').upper()}"
              f"{' ' + g.distro if g.distro else ''}, {len(g.topics)} topics, {len(g.services)} "
              f"non-infrastructure services, {len(g.actions)} actions", file=out)
        if profile is not None:
            try:
                t = select_target(g, profile, None)
            except SelectionError as exc:
                mine = [c for c in exc.candidates if c.profile.id == profile.id][:1]
                for c in mine:
                    report_target(c, out)
                if mine:
                    check_cameras(rb, mine, g, out)
                print(f"FAIL: {exc.reason}", file=out)
                return EXIT_FAIL
            report_target(t, out)
            cams_ok = check_cameras(rb, [t], g, out)
            ok = t.validation.ok and cams_ok
            print("PASS" if ok else "FAIL: an expected camera is not live", file=out)
            return EXIT_OK if ok else EXIT_FAIL
        disc = discover(g)
        failed = False
        for grp in disc.ambiguous:
            print(f"FAIL: ambiguous candidates: {describe_candidates(grp)}", file=out)
            failed = True
        for a, b in disc.dominated:
            print(f"note: {a.label} not reported separately: {b.label} presents all of its "
                  f"required names and more", file=out)
        if not disc.targets and not disc.ambiguous:
            near = "; ".join(f"{t.label} missing {', '.join(m[:6])}" for t, m in disc.near)
            print("FAIL: no supported robot discovered" + (f" (incomplete: {near})" if near else ""),
                  file=out)
            return EXIT_FAIL
        for t in disc.targets:
            report_target(t, out)
            failed |= not t.validation.ok
        valid = [t for t in disc.targets if t.validation.ok]
        if valid and not check_cameras(rb, valid, g, out):
            failed = True
        print("FAIL" if failed else "PASS", file=out)
        return EXIT_FAIL if failed else EXIT_OK
    finally:
        rb.close()


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    return run(args.url, args.expect, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
