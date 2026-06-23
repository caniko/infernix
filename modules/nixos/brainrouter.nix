{
  config,
  lib,
  pkgs,
  infernixBrainrouter,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;

  cfg = config.services.infernix.brainrouter;
  brainrouterModule = infernixBrainrouter.nixosModules.default;
in {
  imports = [brainrouterModule];

  options.services.infernix.brainrouter = {
    enable = mkEnableOption "brainrouter LLM routing proxy via infernix";

    port = mkOption {
      type = types.port;
      default = 9099;
      description = "TCP port for the proxy.";
    };

    listenAddress = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address to bind the TCP listener to.";
    };

    manifest = {
      baseUrl = mkOption {
        type = types.str;
        default = "http://127.0.0.1:3001/v1";
        description = "Base URL of Manifest.";
      };

      apiKeyEnv = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Env var name for Manifest API key.";
      };

      apiKeyFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Path to file containing Manifest API key. Writes it to the environment file.";
      };
    };

    llamaSwap = {
      baseUrl = mkOption {
        type = types.str;
        default = "http://127.0.0.1:8081/v1";
        description = "Base URL of llama-swap.";
      };

      fallbackModel = mkOption {
        type = types.str;
        description = "Model key for fallback from Manifest.";
      };
    };

    modelsPath = mkOption {
      type = types.str;
      default = "/var/lib/models";
      description = "Shared model storage directory.";
    };

    bonsai.modelPath = mkOption {
      type = types.str;
      description = "Path to Bonsai GGUF model file. Use \${models_path} for the models path prefix.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open firewall for brainrouter port.";
    };
  };

  config = mkIf cfg.enable {
    services.brainrouter = {
      enable = true;
      inherit (cfg) port listenAddress;
      inherit (cfg) modelsPath;

      manifest.baseUrl = cfg.manifest.baseUrl;
      manifest.apiKeyEnv = cfg.manifest.apiKeyEnv;

      llamaSwap.baseUrl = cfg.llamaSwap.baseUrl;
      llamaSwap.fallbackModel = cfg.llamaSwap.fallbackModel;

      bonsai.modelPath = cfg.bonsai.modelPath;

      openFirewall = cfg.openFirewall;

      environmentFile = cfg.manifest.apiKeyFile;
    };
  };
}
