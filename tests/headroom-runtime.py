"""Offline production-package qualification against synthetic provider APIs."""

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.request import Request, urlopen


class Provider(BaseHTTPRequestHandler):
    received: ClassVar[list] = []
    disconnected: ClassVar[threading.Event] = threading.Event()

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.received.append((self.path, dict(self.headers), body))
        if body["model"] == "synthetic-cancel":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                while True:
                    self.wfile.write(b"data: " + b" " * 65536 + b'{}\n\n')
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                self.disconnected.set()
            return
        if body.get("stream"):
            data = b'data: {"id":"synthetic","choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
            content_type = "text/event-stream"
        elif self.path.endswith("messages"):
            data = json.dumps({"id": "synthetic", "type": "message", "role": "assistant", "model": body["model"], "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 20, "output_tokens": 1}}).encode()
            content_type = "application/json"
        elif self.path.endswith("responses"):
            data = json.dumps({"id": "synthetic", "object": "response", "status": "completed", "model": body["model"], "output": [], "usage": {"input_tokens": 20, "output_tokens": 1, "total_tokens": 21}}).encode()
            content_type = "application/json"
        elif body["model"] == "synthetic-tool-call":
            data = json.dumps({"id": "synthetic", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "tool_calls": [{"id": "write-call", "type": "function", "function": {"name": "write_file", "arguments": '{"path":"exact.txt","content":"literal \\n text"}'}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}).encode()
            content_type = "application/json"
        else:
            data = json.dumps({"id": "synthetic", "object": "chat.completion", "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 1, "total_tokens": 21}}).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@contextmanager
def proxy_log(directory):
    with tempfile.TemporaryFile(mode="w+", dir=directory) as log:
        yield log


class Runtime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = tempfile.TemporaryDirectory()
        cls.provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
        cls.thread = threading.Thread(target=cls.provider.serve_forever, daemon=True)
        cls.thread.start()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        upstream = f"http://127.0.0.1:{cls.provider.server_port}"
        env = dict(os.environ)
        env.update(HEADROOM_WORKSPACE_DIR=cls.state.name, HEADROOM_CCR_SQLITE_PATH=f"{cls.state.name}/ccr.sqlite", HEADROOM_CCR_BACKEND="sqlite", HEADROOM_CCR_TTL_SECONDS="86400", HEADROOM_BEACON="off", HEADROOM_OFFLINE="1", HEADROOM_DISABLE_KOMPRESS="1", HEADROOM_DISABLE_KOMPRESS_FALLBACK="1", HEADROOM_KOMPRESS_WARMUP="0", HEADROOM_TOOL_SEARCH="0", HEADROOM_OUTPUT_SHAPER="off", HF_HUB_OFFLINE="1", DO_NOT_TRACK="1", OPENAI_API_KEY="synthetic-only", ANTHROPIC_API_KEY="synthetic-only")
        cls.log = cls.enterClassContext(proxy_log(cls.state.name))
        cls.environment = env
        cls.command = [sys.argv[1], "proxy", "--host", "127.0.0.1", "--port", str(cls.port), "--mode", "cache", "--no-cache", "--no-learn", "--no-telemetry", "--openai-api-url", upstream + "/v1", "--anthropic-api-url", upstream]
        try:
            cls.start_proxy()
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def start_proxy(cls):
        cls.process = subprocess.Popen(cls.command, env=cls.environment, stdout=cls.log, stderr=cls.log)
        try:
            for _ in range(120):
                if cls.process.poll() is not None:
                    cls.log.seek(0)
                    raise AssertionError(cls.log.read())
                try:
                    with urlopen(cls.base + "/health", timeout=1) as response:
                        if response.status == 200:
                            return
                except OSError:
                    time.sleep(0.25)
            raise AssertionError("production proxy did not become healthy")
        except BaseException:
            cls.process.terminate()
            cls.process.wait(timeout=15)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=15)
        cls.provider.shutdown()
        cls.provider.server_close()
        cls.state.cleanup()

    def send(self, path, body, extra=None):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer synthetic-client-key", "X-Infernix-Cwd": "/synthetic/project"}
        headers.update(extra or {})
        request = Request(self.base + path, json.dumps(body).encode(), headers)
        with urlopen(request, timeout=30) as response:
            return response.read()

    def test_openai_chat_and_sse(self):
        body = {"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "hello"}]}
        self.assertIn(b"synthetic", self.send("/v1/chat/completions", body))
        path, headers, forwarded = Provider.received[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(forwarded["model"], body["model"])
        self.assertEqual(headers.get("authorization", headers.get("Authorization")), "Bearer synthetic-client-key")
        body["stream"] = True
        self.assertIn(b"[DONE]", self.send("/v1/chat/completions", body))

    def test_responses(self):
        self.assertIn(b"synthetic", self.send("/v1/responses", {"model": "gpt-4.1-mini", "input": "hello"}))
        self.assertEqual(Provider.received[-1][0], "/v1/responses")

    def test_provider_tool_call_is_unchanged(self):
        result = json.loads(self.send("/v1/chat/completions", {"model": "synthetic-tool-call", "messages": [{"role": "user", "content": "Write the requested file."}], "tools": [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {}}}}]}))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"], [{"id": "write-call", "type": "function", "function": {"name": "write_file", "arguments": '{"path":"exact.txt","content":"literal \\n text"}'}}])

    def test_tool_compression_and_cache_prefix(self):
        original = json.dumps([{"index": i, "level": "INFO", "message": "background worker finished", "service": "synthetic"} for i in range(200)])
        tool_call = {"id": "events-call", "type": "function", "function": {"name": "list_events", "arguments": '{"exact": true}'}}
        messages = [{"role": "user", "content": "Inspect the events."}, {"role": "assistant", "content": None, "tool_calls": [tool_call]}, {"role": "tool", "tool_call_id": "events-call", "content": original}]
        body = {"model": "gpt-4.1-mini", "messages": messages, "tools": [{"type": "function", "function": {"name": "list_events", "parameters": {"type": "object", "properties": {}}}}]}
        self.send("/v1/chat/completions", body)
        forwarded = Provider.received[-1][2]["messages"]
        self.assertEqual(forwarded[1]["tool_calls"], [tool_call])
        compressed = forwarded[2]["content"]
        self.assertLess(len(compressed), len(original) * 0.8)
        self.assertEqual(forwarded[2]["tool_call_id"], "events-call")
        # A later turn must replay the identical compressed prefix.
        body["messages"] = [*messages, {"role": "assistant", "content": "Inspected."}, {"role": "user", "content": "Continue."}]
        self.send("/v1/chat/completions", body)
        self.assertEqual(Provider.received[-1][2]["messages"][2]["content"], compressed)

    def test_stream_cancellation(self):
        Provider.disconnected.clear()
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", "/v1/chat/completions", json.dumps({"model": "synthetic-cancel", "stream": True, "messages": [{"role": "user", "content": "hello"}]}), {"Content-Type": "application/json", "Authorization": "Bearer synthetic-client-key"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read(16)
        response.close()
        connection.close()
        self.assertTrue(Provider.disconnected.wait(timeout=10), "cancelled client retained upstream streaming work")

    def test_anthropic_and_cwd(self):
        body = {"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]}
        self.assertIn(b"synthetic", self.send("/v1/messages", body, {"x-api-key": "synthetic-claude-key", "anthropic-version": "2023-06-01"}))
        path, headers, _ = Provider.received[-1]
        self.assertEqual(path, "/v1/messages")
        lowered = {key.lower(): value for key, value in headers.items()}
        self.assertEqual(lowered["x-infernix-cwd"], "/synthetic/project")
        self.assertEqual(lowered["x-api-key"], "synthetic-claude-key")

    def test_native_compression_and_cross_process_retrieval(self):
        rows = [{"index": i, "level": "INFO", "message": "worker completed repetitive background task", "service": "synthetic"} for i in range(200)]
        fatal = "FATAL database disconnected at item 67"
        rows[67].update(level="FATAL", message=fatal)
        original = json.dumps(rows)
        code = """
import json, sys
from headroom.transforms.smart_crusher import SmartCrusher
from headroom.cache.compression_store import CompressionStore
from headroom.cache.backends.sqlite import SQLiteBackend
original = sys.stdin.read()
result = SmartCrusher().crush(original)
store = CompressionStore(backend=SQLiteBackend(sys.argv[1]), default_ttl=86400)
key = store.store(original, result.compressed)
print(json.dumps({'key': key, 'compressed': result.compressed}))
"""
        env = dict(os.environ, HEADROOM_WORKSPACE_DIR=self.state.name, HEADROOM_CCR_SQLITE_PATH=f"{self.state.name}/ccr.sqlite", HEADROOM_BEACON="off")
        process = subprocess.run([sys.executable, "-c", code, f"{self.state.name}/ccr.sqlite"], input=original, capture_output=True, text=True, check=True, env=env)
        result = json.loads(process.stdout)
        self.assertIn(fatal, result["compressed"])
        self.assertLess(len(result["compressed"]), len(original) * 0.8)
        # The writer is gone; resolve its persisted marker through the running
        # proxy without replacing the backend with an in-memory mock.
        with urlopen(self.base + "/v1/retrieve/" + result["key"], timeout=10) as response:
            restored = json.load(response)
        self.assertEqual(restored["original_content"], original)
        # Stop the serving process too: retrieval must survive a fresh proxy
        # process against the same private SQLite state.
        self.process.terminate()
        self.process.wait(timeout=15)
        type(self).start_proxy()
        with urlopen(self.base + "/v1/retrieve/" + result["key"], timeout=10) as response:
            self.assertEqual(json.load(response)["original_content"], original)
        print(f"structural compression: {len(original)} -> {len(result['compressed'])} bytes")


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
