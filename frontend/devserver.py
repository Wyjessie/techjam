"""Static file server for local dev — identical to `python3 -m http.server`
except it sends Cache-Control: no-store on every response.

Why this exists: the plain http.server sends no Cache-Control header at all,
so browsers apply their own heuristic caching against Last-Modified. During
active iteration that means a plain reload can silently serve yesterday's
app.js/style.css with no visible sign anything is wrong — you have to know to
hard-refresh. This removes that footgun so a normal reload always gets what's
actually on disk.

Run (from this directory):
    python3 devserver.py [port]   # defaults to 5500, same as before
"""

import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5500
    HTTPServer(("", port), NoCacheHandler).serve_forever()
