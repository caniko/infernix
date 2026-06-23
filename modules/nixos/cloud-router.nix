{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;

  cfg = config.services.infernix.cloud-router;

  port = 2099;

  providerMap = {
    deepseek = {
      url = "https://api.deepseek.com/v1/chat/completions";
      envVar = "DEEPSEEK_API_KEY";
    };
    xiaomi = {
      url = "https://token-plan-ams.xiaomimimo.com/v1/chat/completions";
      envVar = "XIAOMI_PLATFORM_TOKEN";
    };
    gmi = {
      url = "https://api.gmi-serving.com/v1/chat/completions";
      envVar = "GMI_CLOUD_API_KEY";
    };
  };

  routerPkg = pkgs.writers.writePython3 "infernix-cloud-router" {
    libraries = [pkgs.python3Packages.requests];
    flakeIgnore = ["E501" "E402"];
  } ''
    import http.server
    import json
    import os
    import socketserver
    import requests as http_requests

    PORT = ${toString port}
    PROVIDERS = ${builtins.toJSON providerMap}

    def route_model(model):
        if model.startswith("deepseek-"):
            return PROVIDERS["deepseek"]
        elif model.startswith("mimo-"):
            return PROVIDERS["xiaomi"]
        elif model.startswith("zai-org/"):
            return PROVIDERS["gmi"]
        return PROVIDERS["deepseek"]

    class Proxy(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def do_GET(self):
            if self.path == "/v1/models":
                models = {
                    "object": "list",
                    "data": [
                        {"id": "deepseek-v4-flash", "object": "model"},
                        {"id": "deepseek-v4-pro", "object": "model"},
                        {"id": "mimo-v2-pro", "object": "model"},
                        {"id": "mimo-v2.5-pro", "object": "model"},
                        {"id": "mimo-v2-flash", "object": "model"},
                        {"id": "mimo-v2-omni", "object": "model"},
                        {"id": "zai-org/GLM-5.2-FP8", "object": "model"},
                    ],
                }
                self._json_response(200, models)
            elif self.path == "/api/v1/health":
                self._json_response(200, {"status": "ok"})
            elif self.path == "/health":
                self._json_response(200, {"status": "ok"})
            else:
                self._json_response(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/v1/messages"):
                self._json_response(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length > 0 else {}
                model = body.get("model", "deepseek-v4-flash")
                provider = route_model(model)

                api_key = os.environ.get(provider["envVar"], "")
                if not api_key:
                    self._json_response(500, {"error": f"{provider['envVar']} not set"})
                    return

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                stream = body.get("stream", True)

                if stream:
                    resp = http_requests.post(
                        provider["url"],
                        json=body,
                        headers=headers,
                        stream=True,
                        timeout=300,
                    )
                    self.send_response(resp.status_code)
                    for k, v in resp.headers.items():
                        if k.lower() in ("content-type", "transfer-encoding", "cache-control"):
                            self.send_header(k, v)
                    self.end_headers()
                    for chunk in resp.iter_content(chunk_size=None):
                        if chunk:
                            try:
                                self.wfile.write(chunk)
                                self.wfile.flush()
                            except BrokenPipeError:
                                break
                    resp.close()
                else:
                    resp = http_requests.post(
                        provider["url"],
                        json=body,
                        headers=headers,
                        timeout=300,
                    )
                    self.send_response(resp.status_code)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding",):
                            self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(resp.content)
            except Exception as e:
                self._json_response(500, {"error": str(e)})

        def _json_response(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    with socketserver.TCPServer(("127.0.0.1", PORT), Proxy) as httpd:
        httpd.serve_forever()
  '';

  # Systemd service that renders the env file from agenix secrets
  envFileDir = "/run/infernix-cloud-router";
  envFilePath = "${envFileDir}/env";
in {
  options.services.infernix.cloud-router = {
    enable = mkEnableOption "Minimal cloud LLM router for brainrouter";

    apiKeyFiles = mkOption {
      type = types.attrsOf types.path;
      default = {};
      description = "Attrset of env-var-name → file-path for provider API keys.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open firewall for cloud-router port.";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = [routerPkg];

    # Oneshot that reads agenix-decrypted API keys and writes them as
    # an environment file for the cloud-router service.
    systemd.services.infernix-cloud-router-env = {
      description = "Render cloud-router API key environment";
      before = ["infernix-cloud-router.service"];
      wantedBy = ["infernix-cloud-router.service"];
      path = [pkgs.coreutils];

      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        RuntimeDirectory = "infernix-cloud-router";
        RuntimeDirectoryMode = "0700";
      };

      script = let
        entries = lib.mapAttrsToList (name: path: "export ${name}=$(cat ${path})") cfg.apiKeyFiles;
      in ''
        set -eu
        umask 077
        tmp="${envFilePath}.tmp"
        {
          ${lib.concatStringsSep "\n" entries}
        } > "$tmp"
        chmod 0400 "$tmp"
        mv "$tmp" "${envFilePath}"
      '';
    };

    systemd.services.infernix-cloud-router = {
      description = "Minimal cloud LLM router for brainrouter";
      after = ["network-online.target" "infernix-cloud-router-env.service"];
      wants = ["network-online.target"];
      requires = ["infernix-cloud-router-env.service"];
      wantedBy = ["multi-user.target"];

      serviceConfig = {
        ExecStart = "${routerPkg}";
        Restart = "on-failure";
        RestartSec = 5;
        DynamicUser = true;
        NoNewPrivileges = true;
        PrivateTmp = true;
        EnvironmentFile = [envFilePath];
      };
    };

    networking.firewall = mkIf cfg.openFirewall {
      allowedTCPPorts = [port];
    };
  };
}
