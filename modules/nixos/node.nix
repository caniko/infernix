{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) concatStringsSep mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.node;
  marker = "/run/infernix-node/disabled";
  units = concatStringsSep " " cfg.units;

  nodectl = pkgs.writeShellApplication {
    name = "infernix-nodectl";
    runtimeInputs = [pkgs.systemd];
    text = ''
      set -euo pipefail

      marker="${marker}"
      units=(${units})

      case "''${1:-}" in
        off)
          mkdir -p "$(dirname "$marker")"
          touch "$marker"
          for unit in "''${units[@]}"; do
            systemctl stop "$unit"
          done
          ;;
        on)
          rm -f "$marker"
          for unit in "''${units[@]}"; do
            systemctl start "$unit"
          done
          ;;
        status)
          if [ -e "$marker" ]; then
            echo "off"
            exit 1
          fi
          echo "on"
          ;;
        *)
          echo "usage: infernix-nodectl off|on|status" >&2
          exit 64
          ;;
      esac
    '';
  };

  nodeServer = pkgs.writeText "infernix-node.py" ''
    import http.server
    import json
    import os
    import subprocess
    import socketserver

    marker = "${marker}"
    units = ${builtins.toJSON cfg.units}

    def units_active():
        for unit in units:
            result = subprocess.run(
                ["${pkgs.systemd}/bin/systemctl", "is-active", "--quiet", unit],
                check=False,
            )
            if result.returncode != 0:
                return False
        return True

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            enabled = not os.path.exists(marker)
            active = units_active()
            body = json.dumps({"enabled": enabled, "units_active": active}).encode()
            self.send_response(200 if enabled and active else 503)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    with socketserver.TCPServer(("${cfg.host}", ${toString cfg.port}), Handler) as httpd:
        httpd.serve_forever()
  '';
in {
  options.services.infernix.node = {
    enable = mkEnableOption "Infernix backend node health and drain control";

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Address to bind the node health endpoint to.";
    };

    port = mkOption {
      type = types.port;
      default = 8020;
      description = "Port for the node health endpoint.";
    };

    units = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["llama-swap.service" "ollama.service"];
      description = "Systemd units stopped on `off` and started on `on`.";
    };

    openFirewallInterfaces = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Network interfaces where the node health TCP port is opened.";
    };

    nodectlPackage = mkOption {
      type = types.package;
      readOnly = true;
      default = nodectl;
      description = "Generated infernix-nodectl package for local control hooks.";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = [cfg.nodectlPackage];

    systemd.services.infernix-node = {
      description = "Infernix backend node health endpoint";
      after = ["network-online.target"];
      wants = ["network-online.target"];
      wantedBy = ["multi-user.target"];
      serviceConfig = {
        ExecStart = "${pkgs.python3}/bin/python3 ${nodeServer}";
        RuntimeDirectory = "infernix-node";
        Restart = "on-failure";
        RestartSec = 5;
        DynamicUser = true;
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      };
    };

    networking.firewall.interfaces = lib.genAttrs cfg.openFirewallInterfaces (_: {
      allowedTCPPorts = [cfg.port];
    });
  };
}
