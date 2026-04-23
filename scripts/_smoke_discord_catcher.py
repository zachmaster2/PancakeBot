"""Local HTTP catcher for supervisor smoke test #3.

Runs a single-shot HTTP server on localhost, captures ONE POST, prints the
JSON payload + headers, then exits. Supervisor points its webhook URL at
http://localhost:<port>/catch.
"""
from __future__ import annotations

import http.server
import json
import sys
import threading
import time


class _Catcher(http.server.BaseHTTPRequestHandler):
    captured: dict | None = None
    captured_headers: dict | None = None

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            _Catcher.captured = json.loads(body)
        except Exception:
            _Catcher.captured = {"_raw": body}
        _Catcher.captured_headers = dict(self.headers.items())
        # Return a minimal Discord-like 204 No Content
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        # Silent — suppress default stderr noise.
        pass


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 19876
    server = http.server.HTTPServer(("127.0.0.1", port), _Catcher)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    # Wait up to 15s for one request.
    deadline = time.time() + 15
    while time.time() < deadline and _Catcher.captured is None:
        time.sleep(0.05)
    server.shutdown()
    if _Catcher.captured is None:
        print("NO_REQUEST_CAPTURED")
        return 1
    print("CAPTURED_PAYLOAD:")
    print(json.dumps(_Catcher.captured, indent=2, sort_keys=True))
    print("CAPTURED_HEADERS:")
    print(json.dumps(_Catcher.captured_headers, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
