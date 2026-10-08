import assert from "node:assert/strict";
import { chmodSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { request } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { ensureKeyFile, readKeyFile } from "./credentials.mjs";
import { createProviderServer } from "./server.mjs";

test("credential creation is private, persistent and rejects unsafe files", () => {
  const directory = mkdtempSync(join(tmpdir(), "infernix-auth-"));
  try {
    const path = join(directory, "private", "key");
    const key = ensureKeyFile(path);
    assert.match(key, /^[a-f0-9]{64}$/);
    assert.equal(ensureKeyFile(path), key);
    assert.equal(readKeyFile(path), key);
    chmodSync(path, 0o644);
    assert.throws(() => readKeyFile(path), /private regular file/);
    chmodSync(path, 0o600);
    symlinkSync(path, join(directory, "link"));
    assert.throws(() => readKeyFile(join(directory, "link")));
    writeFileSync(path, "infernix-local");
    assert.throws(() => readKeyFile(path), /256-bit/);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("HTTP authentication precedes body parsing, CWD access and Codex execution", async () => {
  const key = "a".repeat(64);
  const executions = [];
  const server = createProviderServer({ key, execute: async (invocation) => { executions.push(invocation); return "authenticated"; } });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const url = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const authorization of [null, "Bearer infernix-local", `Bearer ${"b".repeat(64)}`]) {
      const headers = { "x-infernix-cwd": "/missing/unauthorized-cwd" };
      if (authorization) headers.authorization = authorization;
      const response = await fetch(`${url}/v1/chat/completions`, { method: "POST", headers, body: "not-json" });
      assert.equal(response.status, 401);
      assert.equal((await response.json()).error.type, "authentication_error");
    }
    // A client withholding its body must still be rejected immediately.
    await new Promise((resolve, reject) => {
      const pending = request(`${url}/v1/chat/completions`, { method: "POST", headers: { "content-length": "100000" }, timeout: 2000 }, (response) => {
        assert.equal(response.statusCode, 401);
        response.resume();
        response.on("end", resolve);
      });
      pending.on("error", reject);
      pending.on("timeout", () => pending.destroy(new Error("authentication waited for the body")));
      pending.flushHeaders();
    });
    assert.equal((await fetch(`${url}/v1/models`)).status, 401);
    assert.equal(executions.length, 0);
    const response = await fetch(`${url}/v1/chat/completions`, {
      method: "POST",
      headers: { authorization: `Bearer ${key}`, "x-infernix-cwd": process.cwd() },
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
    });
    assert.equal(response.status, 200);
    assert.equal((await response.json()).choices[0].message.content, "authenticated");
    assert.equal(executions.length, 1);
    assert.equal(executions[0].cwd, process.cwd());
    assert.equal((await fetch(`${url}/v1/models`, { headers: { authorization: `Bearer ${key}` } })).status, 200);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("a credential is mandatory before opening a listener", () => {
  assert.throws(() => createProviderServer({}), /credential/);
  assert.throws(() => createProviderServer({ key: "infernix-local" }), /credential/);
});
