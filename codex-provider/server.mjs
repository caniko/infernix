import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { createHash, randomUUID, timingSafeEqual } from "node:crypto";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { readKeyFile } from "./credentials.mjs";

const port = process.env.INFERNIX_CODEX_PROVIDER_PORT ? Number(process.env.INFERNIX_CODEX_PROVIDER_PORT) : null;
const host = process.env.INFERNIX_CODEX_PROVIDER_HOST ?? "127.0.0.1";
const codexPath = process.env.CODEX_PATH ?? "codex";
const models = (process.env.INFERNIX_CODEX_MODELS ?? "default").split(",").filter(Boolean);
const maxBodyBytes = 10 * 1024 * 1024;

function textOfContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content.map((part) => {
    if (typeof part === "string") return part;
    if (part.type === "text" || part.type === "input_text") return part.text ?? "";
    if (part.type === "image_url") return `[image: ${part.image_url?.url ?? part.image_url ?? "attached"}]`;
    return `[${part.type ?? "content"}]`;
  }).join("\n");
}

export function promptFromMessages(messages) {
  return messages.map((message) => {
    const role = message.role ?? "user";
    const content = textOfContent(message.content);
    const toolCalls = message.tool_calls?.length
      ? `\nTool calls: ${JSON.stringify(message.tool_calls)}`
      : "";
    return `## ${role}\n${content}${toolCalls}`;
  }).join("\n\n");
}

export function parseCodexJsonl(stdout) {
  const messages = [];
  for (const line of stdout.split("\n")) {
    if (!line.trim()) continue;
    let event;
    try {
      event = JSON.parse(line);
    } catch {
      continue;
    }
    const item = event.item ?? event;
    if ((event.type === "item.completed" || item.type === "agent_message") && item.text) {
      messages.push(item.text);
    }
  }
  return messages.join("\n\n").trim();
}

function requestCwd(request) {
  const candidate = request.headers["x-infernix-cwd"];
  const cwd = typeof candidate === "string" && candidate.length > 0 ? resolve(candidate) : process.cwd();
  if (!existsSync(cwd) || !statSync(cwd).isDirectory()) throw new Error(`working directory is not a directory: ${cwd}`);
  return cwd;
}

function runCodex({ model, prompt, cwd }) {
  const args = ["exec", "--json", "--skip-git-repo-check", "--ask-for-approval", "never", "--sandbox", "workspace-write", "--cd", cwd];
  if (model && model !== "default") args.push("--model", model);
  args.push(prompt);

  return new Promise((resolvePromise, reject) => {
    const child = spawn(codexPath, args, {
      cwd,
      env: { ...process.env, CODEX_PATH: codexPath },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => { stdout += chunk; });
    child.stderr.on("data", (chunk) => { stderr += chunk; });
    child.on("error", reject);
    child.on("close", (code) => {
      const text = parseCodexJsonl(stdout);
      if (code !== 0) reject(new Error(stderr.trim() || `codex exited with status ${code}`));
      else if (!text) reject(new Error("codex returned no final message"));
      else resolvePromise(text);
    });
  });
}

function json(response, status, value) {
  const body = JSON.stringify(value);
  response.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(body) });
  response.end(body);
}

function completion(text, model) {
  return {
    id: `chatcmpl-${randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: "stop" }],
  };
}

async function readBody(request, response) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBodyBytes) {
      json(response, 413, { error: { message: "request body too large", type: "invalid_request_error" } });
      return null;
    }
    chunks.push(chunk);
  }
  try {
    return JSON.parse(Buffer.concat(chunks).toString("utf8"));
  } catch {
    json(response, 400, { error: { message: "request body must be JSON", type: "invalid_request_error" } });
    return null;
  }
}

async function handle(request, response, execute) {
  if (request.method === "GET" && request.url === "/v1/models") {
    return json(response, 200, { object: "list", data: models.map((id) => ({ id, object: "model", owned_by: "codex-cli" })) });
  }
  if (request.method !== "POST" || request.url !== "/v1/chat/completions") return json(response, 404, { error: { message: "not found" } });

  const body = await readBody(request, response);
  if (body == null) return;
  if (!Array.isArray(body.messages) || body.messages.length === 0) return json(response, 400, { error: { message: "messages must be a non-empty array", type: "invalid_request_error" } });

  try {
    const requestedModel = typeof body.model === "string" ? body.model : "default";
    const model = requestedModel.includes(",") ? requestedModel.split(",").pop() : requestedModel;
    const text = await execute({ model, prompt: promptFromMessages(body.messages), cwd: requestCwd(request) });
    const result = completion(text, model);
    if (!body.stream) return json(response, 200, result);
    response.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    response.write(`data: ${JSON.stringify({ ...result, choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: "stop" }] })}\n\n`);
    response.end("data: [DONE]\n\n");
  } catch (error) {
    json(response, 502, { error: { message: error.message, type: "upstream_error" } });
  }
}

function selfTest() {
  assert.match(promptFromMessages([{ role: "user", content: "hello" }]), /## user\nhello/);
  assert.equal(parseCodexJsonl('{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'), "done");
  assert.equal(parseCodexJsonl("not-json\n"), "");
}

export function createProviderServer({ key, execute = runCodex }) {
  if (typeof key !== "string" || !/^[a-f0-9]{64}$/.test(key)) throw new Error("a private Codex provider credential is required");
  const digest = (value) => createHash("sha256").update(value).digest();
  const expected = digest(`Bearer ${key}`);
  return createServer((request, response) => {
    // Loopback TCP is shared by local accounts. Reject before reading the body
    // or inspecting the caller-selected workspace, including on discovery.
    const authorization = request.headers.authorization;
    if (typeof authorization !== "string" || !timingSafeEqual(expected, digest(authorization))) {
      response.setHeader("connection", "close");
      response.setHeader("www-authenticate", "Bearer");
      return json(response, 401, { error: { message: "invalid provider credential", type: "authentication_error" } });
    }
    handle(request, response, execute).catch((error) => json(response, 500, { error: { message: error.message } }));
  });
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.includes("--self-test")) selfTest();
  else {
    if (port == null) throw new Error("INFERNIX_CODEX_PROVIDER_PORT is required");
    if (!process.env.INFERNIX_CODEX_PROVIDER_KEY_FILE) throw new Error("INFERNIX_CODEX_PROVIDER_KEY_FILE is required");
    createProviderServer({ key: readKeyFile(process.env.INFERNIX_CODEX_PROVIDER_KEY_FILE) }).listen(port, host);
  }
}
