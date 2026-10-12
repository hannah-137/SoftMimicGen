"""Small web server for a share folder. It supports byte ranges, so the browser can seek in the videos.
`share.py` copies this file into the share folder as `serve.py`.

Usage:
    python3 serve.py [port]        # default port 8766, open http://localhost:8766

The server listens on this machine only (127.0.0.1). Forward the port to open the page from another computer.
"""
import functools
import http.server
import os
import re
import sys


class Handler(http.server.SimpleHTTPRequestHandler):
    span = None  # (first byte, last byte) of the current range request

    def send_head(self):
        self.span = None
        path = self.translate_path(self.path)
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "").strip())
        if not m or not (m.group(1) or m.group(2)) or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        if m.group(1):
            first, last = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
        else:  # "bytes=-N": the last N bytes
            first, last = max(size - int(m.group(2)), 0), size - 1
        last = min(last, size - 1)
        if first > last:
            self.send_error(416, "Requested Range Not Satisfiable")
            return None
        f = open(path, "rb")
        f.seek(first)
        self.span = (first, last)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        if self.span is None:
            return super().copyfile(source, outputfile)
        left = self.span[1] - self.span[0] + 1
        while left > 0:
            chunk = source.read(min(256 * 1024, left))
            if not chunk:
                break
            outputfile.write(chunk)
            left -= len(chunk)

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser stopped a video download: normal when a video leaves the screen

    def log_message(self, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    folder = os.path.dirname(os.path.abspath(__file__))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=folder))
    print(f"serving {folder} at http://localhost:{port}  (Ctrl+C to stop)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
