import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout } from "node:timers/promises";
import { test } from "node:test";

test("the pinned CCR authenticates with the private runtime key, never a public placeholder", { timeout: 60000 }, async () => {
  const home = mkdtempSync(join(tmpdir(), "infernix-ccr-auth-"));
  const reservation = createServer();
  await new Promise((resolve) => reservation.listen(0, "127.0.0.1", resolve));
  const port = reservation.address().port;
  await new Promise((resolve) => reservation.close(resolve));
  const key = "a".repeat(64);
  const config = JSON.parse(readFileSync(process.env.CCR_CONFIG_TEMPLATE, "utf8"));
  config.PORT = port;
  mkdirSync(join(home, ".claude-code-router"), { mode: 0o700 });
  writeFileSync(join(home, ".claude-code-router", "config.json"), JSON.stringify(config), { mode: 0o600 });
  const env = { ...process.env, HOME: home, XDG_CONFIG_HOME: join(home, ".config"), XDG_DATA_HOME: join(home, ".local/share"), INFERNIX_CODEX_PROVIDER_API_KEY: key };
  const child = spawn(process.env.CCR_PATH, ["start"], { env, detached: true, stdio: ["ignore", "pipe", "pipe"] });
  let output = "";
  child.stdout.on("data", (chunk) => { output += chunk; });
  child.stderr.on("data", (chunk) => { output += chunk; });
  const exited = new Promise((resolve) => child.on("exit", resolve));
  const url = `http://127.0.0.1:${port}/v1/messages`;
  const send = (token) => fetch(url, { method: "POST", headers: { "content-type": "application/json", ...(token ? { authorization: `Bearer ${token}` } : {}) }, body: "not-json", signal: AbortSignal.timeout(2000) });
  try {
    const deadline = Date.now() + 40000;
    let ready = false;
    while (Date.now() < deadline) {
      if (child.exitCode !== null) throw new Error(`CCR exited before listening: ${output}`);
      try {
        const response = await send();
        await response.text();
        ready = true;
        break;
      } catch {
        await setTimeout(100);
      }
    }
    assert.ok(ready, `CCR did not listen: ${output}`);
    const serverPid = Number(readFileSync(join(home, ".claude-code-router", ".claude-code-router.pid"), "utf8").trim());
    assert.equal(serverPid, child.pid, "the supervised CCR process must own the listener, without a background child");
    for (const token of [null, "infernix-local", "$INFERNIX_CODEX_PROVIDER_API_KEY", "b".repeat(64)]) {
      // Valid JSON ensures the authorization middleware, not a parse failure,
      // rejects the unauthenticated request.
      const response = await fetch(url, { method: "POST", headers: token ? { authorization: `Bearer ${token}`, "content-type": "application/json" } : { "content-type": "application/json" }, body: "{}", signal: AbortSignal.timeout(2000) });
      assert.equal(response.status, 401, `CCR accepted a public/missing credential: ${await response.text()}`);
    }
    const authenticated = await send(key);
    const body = await authenticated.text();
    assert.equal(authenticated.status, 400, `authenticated malformed request did not reach parsing: ${body}`);
  } finally {
    if (child.exitCode === null) {
      process.kill(-child.pid, "SIGTERM");
      await exited;
    }
    rmSync(home, { recursive: true, force: true });
  }
});
