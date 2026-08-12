import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";

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

async function handle(request, response) {
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
    const text = await runCodex({ model, prompt: promptFromMessages(body.messages), cwd: requestCwd(request) });
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

if (process.argv.includes("--self-test")) selfTest();
else if (port == null) throw new Error("INFERNIX_CODEX_PROVIDER_PORT is required");
else createServer((request, response) => handle(request, response).catch((error) => json(response, 500, { error: { message: error.message } }))).listen(port, host);
