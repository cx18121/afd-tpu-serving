#!/usr/bin/env python3
"""Stand-in for `vllm serve` that needs no TPU, for testing the scripts.

Speaks just enough of the OpenAI API for vm_baseline.sh: /health,
/v1/chat/completions (the sanity check), streaming /v1/completions
(`vllm bench serve --backend vllm`), and /start_profile, /stop_profile.

MOCK_GARBAGE=1 makes chat replies wrong, to test that the sanity check fails.
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_PORT", "8000"))
TOKEN_DELAY_S = float(os.environ.get("MOCK_TOKEN_DELAY_S", "0.002"))
GARBAGE = os.environ.get("MOCK_GARBAGE") == "1"
ANSWERS = {
    "capital of France": "Paris",
    "17 * 23": "391",
    "Count from 1 to 30": ", ".join(str(i) for i in range(1, 31)),
    "sky is blue": "Sunlight scatters off air molecules; blue light scatters most.",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("mock: " + fmt % args + "\n")

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "mock", "object": "model"}]})
        else:
            self._json(404, {"error": self.path})

    def do_POST(self):
        if self.path in ("/start_profile", "/stop_profile"):
            self._body()
            self._json(200, {})
        elif self.path == "/v1/chat/completions":
            req = self._body()
            prompt = req["messages"][-1]["content"]
            reply = next((a for k, a in ANSWERS.items() if k in prompt), "ok")
            if GARBAGE:
                reply = "!!!!!!!!"
            self._json(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": reply}}]})
        elif self.path == "/v1/completions":
            self._stream(self._body())
        else:
            self._json(404, {"error": self.path})

    def _stream(self, req):
        n = int(req.get("max_tokens") or 16)
        prompt = req.get("prompt") or ""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        for _ in range(n):
            time.sleep(TOKEN_DELAY_S)
            send({"choices": [{"index": 0, "text": " a", "finish_reason": None}]})
        send({"choices": [], "usage": {"prompt_tokens": len(str(prompt).split()),
                                       "completion_tokens": n, "total_tokens": n}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    print(f"mock server on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
