"""Offline production-package qualification against synthetic provider APIs."""

import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect
from websockets.sync.server import serve


class Provider(BaseHTTPRequestHandler):
    received: ClassVar[list] = []
    disconnected: ClassVar[threading.Event] = threading.Event()

    def log_message(self, *_args):
        pass

    def do_GET(self):
        self.received.append((self.path, dict(self.headers), None))
        self.send_error(501)

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
            if self.path.endswith("responses"):
                data = b'data: {"type":"response.completed","response":{"id":"synthetic","status":"completed","output":[]}}\n\n'
            else:
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


class WebsocketPolicy(unittest.TestCase):
    def test_validated_address_and_hostname_identity(self):
        from headroom.proxy.upstream_guard import UpstreamPin
        from headroom.proxy.upstream_pinning import connect_guarded_websocket

        pin = UpstreamPin(("8.8.8.8", "1.1.1.1"), time.monotonic() + 60)
        with patch("headroom.proxy.upstream_pinning.guarded_pin", return_value=pin), patch("websockets.proxy.get_proxy", return_value=None):
            for attempt, address in enumerate(pin.addresses):
                connection = connect_guarded_websocket("wss://provider.example/v1/responses", upstream_base_url="https://provider.example/v1", attempt=attempt)
                self.assertEqual(connection.create_connection_kwargs["host"], address)
                self.assertIsNone(connection.proxy)
                self.assertEqual(connection.ws_uri.host, "provider.example")
                redirect = RuntimeError("synthetic redirect")
                self.assertIs(connection.process_redirect(redirect), redirect)

    def test_expired_or_unpinnable_destination_is_refused(self):
        from headroom.proxy.upstream_guard import UpstreamPin
        from headroom.proxy.upstream_pinning import (
            UnpinnableUpstreamError,
            connect_guarded_websocket,
        )

        for deadline, proxy in [(time.monotonic() - 1, None), (time.monotonic() + 60, "http://proxy.example:8080")]:
            with self.subTest(deadline=deadline, proxy=proxy), patch("headroom.proxy.upstream_pinning.guarded_pin", return_value=UpstreamPin(("8.8.8.8",), deadline)), patch("websockets.proxy.get_proxy", return_value=proxy), patch("headroom.proxy.upstream_pinning.proxied_upstream_listed", return_value=False), self.assertRaises(UnpinnableUpstreamError):
                connect_guarded_websocket("wss://provider.example/v1/responses", upstream_base_url="https://provider.example/v1")


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
        started = time.monotonic()
        cls.process = subprocess.Popen(cls.command, env=cls.environment, stdout=cls.log, stderr=cls.log)
        try:
            # Full-flake CI also builds Rust closures. Bound cold-import time
            # using elapsed time, and keep failing immediately on a crash.
            while time.monotonic() - started < 90:
                if cls.process.poll() is not None:
                    cls.log.seek(0)
                    raise AssertionError(cls.log.read())
                try:
                    with urlopen(cls.base + "/health", timeout=1) as response:
                        if response.status == 200:
                            print(f"proxy ready after {time.monotonic() - started:.1f}s", flush=True)
                            return
                except OSError:
                    time.sleep(0.25)
            cls.log.seek(0)
            raise AssertionError(f"production proxy did not become healthy after {time.monotonic() - started:.1f}s\n{cls.log.read()[-8192:]}")
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

    def test_disallowed_http_upstream_does_not_receive_credentials(self):
        before = len(Provider.received)
        for denied in ["http://127.0.0.1:1/v1", "not-a-url", "http://127.0.0.1:bad/v1", "http://user:password@127.0.0.1/v1"]:
            with self.subTest(denied=denied):
                with self.assertRaises(HTTPError) as failure:
                    self.send("/v1/responses", {"model": "gpt-4.1-mini", "input": "synthetic denial"}, {"X-Headroom-Base-Url": denied, "Authorization": "Bearer synthetic-denied"})
                self.assertEqual(failure.exception.code, 403)
        self.assertEqual(len(Provider.received), before, "rejected credentials reached the default upstream")

    def test_custom_websocket_upstreams_are_request_local(self):
        received = [[], []]

        def handler(index):
            def receive(connection):
                body = json.loads(connection.recv(timeout=10))
                received[index].append((connection.request.path, dict(connection.request.headers), body))
                connection.send(json.dumps({"type": "response.completed", "response": {"id": f"synthetic-{index}", "status": "completed", "output": []}}))
            return receive

        with serve(handler(0), "127.0.0.1", 0) as first, serve(handler(1), "127.0.0.1", 0) as second:
            threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in [first, second]]
            for thread in threads:
                thread.start()
            urls = [f"http://127.0.0.1:{server.socket.getsockname()[1]}/custom/v1" for server in [first, second]]
            previous = self.environment.get("HEADROOM_ALLOWED_BASE_URLS")
            self.environment["HEADROOM_ALLOWED_BASE_URLS"] = ",".join(urls)
            self.process.terminate()
            self.process.wait(timeout=15)
            type(self).start_proxy()
            before = len(Provider.received)
            bodies = [{"type": "response.create", "model": "gpt-4.1-mini", "input": [{"type": "function_call_output", "call_id": f"call-{index}", "output": "exact synthetic tool result"}]} for index in range(2)]

            def request(index):
                headers = {"Authorization": f"Bearer synthetic-custom-{index}", "X-Headroom-Base-Url": urls[index], "X-Headroom-Original-Path": "/responses"}
                with connect(self.base.replace("http://", "ws://") + "/v1/responses", additional_headers=headers, proxy=None, open_timeout=10) as connection:
                    connection.send(json.dumps(bodies[index]))
                    self.assertEqual(json.loads(connection.recv(timeout=10))["response"]["id"], f"synthetic-{index}")

            try:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    list(executor.map(request, range(2)))
                self.assertEqual(len(Provider.received), before, "custom credentials reached the static upstream")
                for index in range(2):
                    self.assertEqual(len(received[index]), 1)
                    path, headers, body = received[index][0]
                    self.assertEqual(path, "/custom/v1/responses")
                    self.assertEqual(headers["authorization"], f"Bearer synthetic-custom-{index}")
                    self.assertFalse(any(key.lower().startswith("x-headroom-") for key in headers))
                    self.assertEqual(body["model"], bodies[index]["model"])
                    self.assertEqual(body["input"], bodies[index]["input"])
                for denied in ["http://127.0.0.1:1/v1", "https://example.invalid/v1", "not-a-url", "http://127.0.0.1:bad/v1"]:
                    with self.subTest(denied=denied), self.assertRaises(InvalidStatus), connect(self.base.replace("http://", "ws://") + "/v1/responses", additional_headers={"Authorization": "Bearer synthetic-denied", "X-Headroom-Base-Url": denied}, proxy=None, open_timeout=10):
                        self.fail("disallowed custom upstream was accepted")
                self.assertEqual(len(Provider.received), before, "rejected credentials reached the static upstream")
            finally:
                if previous is None:
                    self.environment.pop("HEADROOM_ALLOWED_BASE_URLS", None)
                else:
                    self.environment["HEADROOM_ALLOWED_BASE_URLS"] = previous
                self.process.terminate()
                self.process.wait(timeout=15)
                type(self).start_proxy()

    def test_provider_tool_call_is_unchanged(self):
        result = json.loads(self.send("/v1/chat/completions", {"model": "synthetic-tool-call", "messages": [{"role": "user", "content": "Write the requested file."}], "tools": [{"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {}}}}]}))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(choice["message"]["tool_calls"], [{"id": "write-call", "type": "function", "function": {"name": "write_file", "arguments": '{"path":"exact.txt","content":"literal \\n text"}'}}])

    def test_custom_websocket_http_fallback_keeps_selected_provider(self):
        class CustomProvider(Provider):
            received: ClassVar[list] = []

        provider = ThreadingHTTPServer(("127.0.0.1", 0), CustomProvider)
        thread = threading.Thread(target=provider.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{provider.server_port}/custom/v1"
        previous = self.environment.get("HEADROOM_ALLOWED_BASE_URLS")
        self.environment["HEADROOM_ALLOWED_BASE_URLS"] = url
        self.process.terminate()
        self.process.wait(timeout=15)
        type(self).start_proxy()
        before = len(Provider.received)
        try:
            headers = {"Authorization": "Bearer synthetic-fallback-key", "X-Headroom-Base-Url": url, "X-Headroom-Original-Path": "/responses"}
            with connect(self.base.replace("http://", "ws://") + "/v1/responses", additional_headers=headers, proxy=None, open_timeout=10) as connection:
                connection.send(json.dumps({"type": "response.create", "model": "gpt-4.1-mini", "input": "synthetic fallback"}))
                self.assertEqual(json.loads(connection.recv(timeout=15))["response"]["id"], "synthetic")
            self.assertEqual(len(Provider.received), before)
            path, headers, body = CustomProvider.received[-1]
            self.assertEqual(path, "/custom/v1/responses")
            self.assertEqual({key.lower(): value for key, value in headers.items()}["authorization"], "Bearer synthetic-fallback-key")
            self.assertEqual(body["input"], "synthetic fallback")
        finally:
            provider.shutdown()
            provider.server_close()
            if previous is None:
                self.environment.pop("HEADROOM_ALLOWED_BASE_URLS", None)
            else:
                self.environment["HEADROOM_ALLOWED_BASE_URLS"] = previous
            self.process.terminate()
            self.process.wait(timeout=15)
            type(self).start_proxy()

    def test_responses_stream_cancellation(self):
        Provider.disconnected.clear()
        connection = HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.request("POST", "/v1/responses", json.dumps({"model": "synthetic-cancel", "stream": True, "input": "synthetic cancellation"}), {"Content-Type": "application/json", "Authorization": "Bearer synthetic-client-key"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        response.read(16)
        response.close()
        connection.close()
        self.assertTrue(Provider.disconnected.wait(timeout=10), "cancelled Responses client retained upstream work")

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

    def test_proxy_marker_retrieval_through_mcp_after_restart(self):
        original = json.dumps([f"synthetic worker {i}: " + "background task " * 50 for i in range(250)])
        messages = [{"role": "user", "content": "Inspect the synthetic events."}, {"role": "assistant", "content": None, "tool_calls": [{"id": "mcp-call", "type": "function", "function": {"name": "list_events", "arguments": "{}"}}]}, {"role": "tool", "tool_call_id": "mcp-call", "content": original}]
        self.send("/v1/chat/completions", {"model": "gpt-4.1-mini", "messages": messages})
        compressed = Provider.received[-1][2]["messages"][2]["content"]
        self.assertLess(len(compressed), len(original) * 0.8)
        marker = re.search(r"(?:hash=|<<ccr:)([a-f0-9]+)", compressed)
        self.assertIsNotNone(marker, "proxy did not emit a retrieval marker")
        self.process.terminate()
        self.process.wait(timeout=15)
        type(self).start_proxy()
        code = '''
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
async def retrieve():
    params = StdioServerParameters(command=sys.argv[1], args=["mcp", "serve", "--proxy-url", sys.argv[2]], env=dict(os.environ))
    async with stdio_client(params) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert any(tool.name == "headroom_retrieve" for tool in tools.tools)
            result = await session.call_tool("headroom_retrieve", {"hash": sys.argv[3]})
            assert not result.isError
            print(result.content[0].text)
asyncio.run(asyncio.wait_for(retrieve(), timeout=30))
'''
        result = subprocess.run([sys.executable, "-c", code, self.command[0], self.base, marker.group(1)], env=self.environment, capture_output=True, text=True, timeout=40, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        # The proxy canonicalizes JSON before storing it; every original value,
        # including whitespace inside strings, must survive retrieval.
        self.assertEqual(json.loads(json.loads(result.stdout)["original_content"]), json.loads(original))

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
    unittest.main(argv=[sys.argv[0], *sys.argv[2:]], verbosity=2)
