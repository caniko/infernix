# Headroom

The production `headroom` package is an immutable Python 3.12 environment made
with harbor-py. `uv.lock` pins Headroom 0.40.0 and all proxy dependencies,
including its native compression extension. No pip installation or model
download happens when a service starts.

`codex-routing.patch` repairs the 0.40.0 session adapter and Responses
WebSocket path. Custom providers keep their transport setting and API prefix;
the selected destination stays request-local during WebSocket-to-HTTP fallback.
Explicit invalid destinations are rejected before credentials are forwarded.
WebSockets retain validated address pinning, Host/TLS identity, and refuse
redirects. Retire this patch only after the replacement package passes the
custom-provider routing, fallback, cancellation and MCP retrieval fixtures.

Refresh deliberately with `uv lock --upgrade` in this directory and build
`canix cache binary build .#headroom` from the Infernix checkout. The lock must
remain compatible with Python 3.12 and the package's native library overrides.

Import `infernix.homeModules.headroom` (also included by `homeModules.default`),
set `services.infernix.headroom.enable`, and declare named `instances` with a
port and their existing `openaiUrl` or `anthropicUrl`. The latter is an origin;
the former includes the upstream API prefix. Instances bind to 127.0.0.1.

Each instance exports `endpoints.<name>.{url,healthUrl,unit}`. Upstream units go
in `instances.<name>.requires`; readiness is checked before dependent services
start. State is private under `xdg.stateHome/infernix-headroom`. Instances share
an SQLite CCR database, retained for 24 hours by default, so any retrieval
proxy can resolve a marker from another instance after a restart.

Set `mcp.enable` and select `mcp.instance` to register retrieval through Fleetix
for OpenCode, Codex and Claude. This uses the proxy API; upstream's deprecated
`--direct` mode is intentionally unused.

Defaults preserve prefix-cache stability, disable semantic response caching,
ML compression/downloads, traffic learning, tool search, output shaping and
external telemetry. Existing provider authentication stays on the request.
Non-secret changes belong in `instances.<name>.environment`; runtime
credentials may use `environmentFile`. Credentials never belong in Nix data.

Codex OAuth routing is owned by Headroom and goes to ChatGPT directly, even
when `openaiUrl` is configured. Put Jev before Headroom where Jev is required.
Custom Codex providers use upstream's session-only adapter and may require an
explicit `HEADROOM_ALLOWED_BASE_URLS` allowlist for local provider origins.
