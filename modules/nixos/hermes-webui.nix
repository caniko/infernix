{
  config,
  lib,
  infernixHermesAgent,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.hermes-webui;
in {
  options.services.infernix.hermes-webui = {
    enable = mkEnableOption "Hermes WebUI via Infernix";

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address the WebUI binds to.";
    };

    port = mkOption {
      type = types.port;
      default = 8787;
      description = "TCP port the WebUI listens on.";
    };

    passwordFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "File containing the WebUI password.";
    };

    stateDir = mkOption {
      type = types.str;
      default = "/var/lib/hermes-webui";
      description = "State directory for sessions and settings.";
    };

    extraEnv = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Extra environment variables for the service.";
    };

    user = mkOption {
      type = types.str;
      default = "hermes-webui";
      description = "System user the service runs as.";
    };

    group = mkOption {
      type = types.str;
      default = "hermes-webui";
      description = "System group for the service user.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open the firewall for the WebUI port.";
    };
  };

  config = mkIf cfg.enable {
    services.hermes-webui = {
      enable = true;
      inherit (cfg) host port passwordFile stateDir extraEnv user group openFirewall;
      # Auto-wire agentDir from the hermes-agent flake input source.
      agentDir = infernixHermesAgent;
    };
  };
}
