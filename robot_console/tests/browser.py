"""A headless Chromium driven over the DevTools protocol, for the view page's tests.

No Playwright and no Selenium: the console's `dev` extra already carries `websockets`,
which is all the DevTools protocol needs, and any Chromium-family browser on the machine
speaks it. `find_browser()` looks at `$VIEW_TEST_BROWSER` first, then the usual install
locations; the tests skip, saying so, when there is none.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

from websockets.sync.client import connect

_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
    "microsoft-edge", "brave-browser",
)


def find_browser() -> Optional[str]:
    explicit = os.environ.get("VIEW_TEST_BROWSER")
    if explicit:
        return explicit
    for candidate in _CANDIDATES:
        if os.path.isabs(candidate):
            if os.access(candidate, os.X_OK):
                return candidate
        elif shutil.which(candidate):
            return shutil.which(candidate)
    return None


class Browser:
    """One headless browser with one page, and `evaluate` on it."""

    def __init__(self, executable: str) -> None:
        self._profile = tempfile.mkdtemp(prefix="view-test-")
        self._proc = subprocess.Popen(
            [executable, "--headless=new", "--remote-debugging-port=0",
             f"--user-data-dir={self._profile}", "--no-first-run", "--no-default-browser-check",
             "--disable-extensions", "--disable-background-networking",
             "--window-size=1200,1600", "about:blank"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        port_file = Path(self._profile, "DevToolsActivePort")
        deadline = time.monotonic() + 20
        while not (port_file.exists() and port_file.read_text().strip()):
            if time.monotonic() > deadline or self._proc.poll() is not None:
                self.close()
                raise RuntimeError(f"{executable} did not start its DevTools endpoint")
            time.sleep(0.05)
        port = int(port_file.read_text().split()[0])
        targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list"))
        page = next(t for t in targets if t.get("type") == "page")
        self._ws = connect(page["webSocketDebuggerUrl"], max_size=None).__enter__()
        self._ids = itertools.count(1)
        self.call("Page.enable")
        self.call("Runtime.enable")

    def call(self, method: str, params: Optional[dict] = None, timeout: float = 15.0) -> dict:
        call_id = next(self._ids)
        self._ws.send(json.dumps({"id": call_id, "method": method, "params": params or {}}))
        deadline = time.monotonic() + timeout
        while True:
            message = json.loads(self._ws.recv(timeout=max(0.01, deadline - time.monotonic())))
            if message.get("id") == call_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def navigate(self, url: str) -> None:
        # Navigating to the URL already shown never answers in headless Chromium.
        if url == "about:blank" and self.evaluate("location.href") == url:
            return
        self.call("Page.navigate", {"url": url})

    def evaluate(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", {"expression": expression, "awaitPromise": True,
                                                "returnByValue": True})
        if "exceptionDetails" in result:
            raise RuntimeError(f"page raised: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, timeout: float = 10.0) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            try:
                value = self.evaluate(expression)
            except RuntimeError:
                value = None
            if value:
                return value
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {expression}")
            time.sleep(0.05)

    def screenshot(self, path: str) -> None:
        import base64

        data = self.call("Page.captureScreenshot", {"format": "png",
                                                    "captureBeyondViewport": True})["data"]
        Path(path).write_bytes(base64.b64decode(data))

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        shutil.rmtree(self._profile, ignore_errors=True)
