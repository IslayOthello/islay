#!/usr/bin/env python3
"""
Islay Desktop Web GUI Server (Cross-platform Fallback)
Serves gui/web and bridges HTTP POST /api to `luajit gui/bridge.lua`
"""
import http.server
import json
import os
import subprocess
import sys
import threading
import webbrowser

PORT = 8080
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WEB_DIR = os.path.join(ROOT_DIR, "gui", "web")

class LuaBridge:
    def __init__(self):
        luajit_bin = "luajit"
        for candidate in ["/opt/homebrew/bin/luajit", "/usr/local/bin/luajit", "luajit", "lua"]:
            if subprocess.run(["which", candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                luajit_bin = candidate
                break

        self.proc = subprocess.Popen(
            [luajit_bin, "gui/bridge.lua", "pipe"],
            cwd=ROOT_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        self.lock = threading.Lock()

    def query(self, req_json):
        with self.lock:
            try:
                line = json.dumps(req_json) + "\n"
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
                res = self.proc.stdout.readline()
                if res:
                    return json.loads(res.strip())
            except Exception as e:
                print("Bridge error:", e, file=sys.stderr)
        return {"error": "bridge failure"}

bridge = LuaBridge()

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_POST(self):
        if self.path == "/api":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8")
            try:
                req = json.loads(body)
                res = bridge.query(req)
                res_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                self.send_response(500)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

def main():
    print(f"==> Islay Desktop Web GUI running at http://localhost:{PORT}")
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Timer(0.5, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExiting server...")

if __name__ == "__main__":
    main()
