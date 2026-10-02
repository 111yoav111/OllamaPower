"""Fake ``llama-server`` and ``ggml-rpc-server`` programs, for hermetic split-mode tests.

``install_fakes(dir, devices)`` writes two small Python scripts under the real
program names. They answer ``--version`` (the pinned build) and
``--list-devices`` like llama.cpp does, and:

* the fake rpc-server listens where it's told and echoes bytes back;
* the fake llama-server records its arguments to ``args.json``, checks it can
  reach every ``--rpc`` endpoint (an echo round trip through the agent's
  relay), then serves ``/health`` and an OpenAI-style streaming
  ``/v1/chat/completions`` with llama-server's ``timings``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RPC = r"""
import socket, sys, threading
args = sys.argv[1:]
if "--version" in args:
    print("version: 0.5.0-dev (build BUILD, commit fake)"); sys.exit(0)
port = int(args[args.index("-p") + 1]); host = args[args.index("-H") + 1]
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind((host, port)); srv.listen(8)
print("Starting RPC server", flush=True)
def serve(c):
    with c:
        while data := c.recv(65536):
            c.sendall(data)
while True:
    c, _ = srv.accept()
    print("Accepted client connection", flush=True)
    threading.Thread(target=serve, args=(c,), daemon=True).start()
"""

SERVER = r"""
import json, os, socket, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
args = sys.argv[1:]
HERE = Path(__file__).parent
if "--version" in args:
    print("version: 0.5.0-dev (build BUILD, commit fake)"); sys.exit(0)
if "--list-devices" in args:
    print("Available devices:")
    for line in DEVICES:
        print("  " + line)
    if not DEVICES:
        print("  (none)")
    sys.exit(0)
(HERE / "args.json").write_text(json.dumps(args))
if "--rpc" in args:
    for endpoint in args[args.index("--rpc") + 1].split(","):
        host, port = endpoint.rsplit(":", 1)
        with socket.create_connection((host.strip("[]"), int(port)), timeout=5) as s:
            s.sendall(b"ping"); assert s.recv(4) == b"ping"
if (HERE / "fail_start").exists():
    print("error: failed to load model", flush=True); sys.exit(3)
port = int(args[args.index("--port") + 1])

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(b'{"status":"ok"}')
    def do_POST(self):
        if self.headers.get("authorization") != "Bearer " + os.environ.get("LLAMA_API_KEY", "-"):
            self.send_response(401); self.end_headers(); return
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        (HERE / "last_request.json").write_text(json.dumps({"path": self.path, "body": body}))
        if body.get("messages") and body["messages"][-1].get("content") == "fail":
            self.send_response(400); self.send_header("content-type", "application/json"); self.end_headers()
            self.wfile.write(json.dumps({"error": {"code": 400, "message": "bad request from fake"}}).encode()); return
        last = body["messages"][-1].get("content") if body.get("messages") else ""
        if last == "hang":
            time.sleep(3600)  # never answers: the hub's stall timeout must notice
        self.send_response(200); self.send_header("content-type", "text/event-stream"); self.end_headers()
        def ev(obj): self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()
        for word in ["Hello", " from", " split"]:
            ev({"choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]})
            if last == "die":
                print("GGML_ASSERT failed: fake crash", flush=True)
                os._exit(1)
            time.sleep(float(body.get("fake_delay", 0)))
        if body.get("tools"):
            for fn in ({"name": "get", "arguments": '{"a":'}, {"arguments": ' 1}'}):
                ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": fn}]}}]})
        ev({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        timings = {"prompt_n": 7, "prompt_ms": 70.0, "predicted_n": 3, "predicted_ms": 150.0}
        if (HERE / "timings.json").exists():  # tests set how fast "split" is
            timings.update(json.loads((HERE / "timings.json").read_text()))
        ev({"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}, "timings": timings})
        self.wfile.write(b"data: [DONE]\n\n")

print("listening", flush=True)
ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def install_fakes(directory: Path, devices: list[str], build: int) -> Path:
    """Write fake llama.cpp programs into ``directory``; ``devices`` are --list-devices lines."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, source in (("ggml-rpc-server", RPC), ("llama-server", SERVER)):
        body = source.replace("BUILD", str(build)).replace("DEVICES", json.dumps(devices))
        path = directory / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)
    return directory
