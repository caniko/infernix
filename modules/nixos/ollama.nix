{
  config,
  lib,
  infernisBleedingNixpkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types;
  gpuCfg = config.services.infernis.gpu;
  cfg = config.services.infernis.ollama;
  gpuLib = import ../../lib/gpu.nix {inherit lib;};
in {
  options.services.infernis.ollama = {
    enable = mkEnableOption "Ollama model serving";

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Address to bind Ollama to.";
    };

    port = mkOption {
      type = types.port;
      default = 11434;
      description = "Ollama API port.";
    };

    loadModels = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["gemma4:31b-it-q4_K_M"];
      description = "Models to preload on service start.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to open firewall ports for Ollama.";
    };

    extraConfig = mkOption {
      type = types.attrs;
      default = {};
      description = "Extra attrs merged into services.ollama for options not covered by this module.";
    };
  };

  config = mkIf cfg.enable (let
    # Re-import nixpkgs master with the consumer's GPU config so
    # ollama-rocm / ollama-cuda inherit rocmSupport / cudaSupport.
    bleedingPkgs = gpuLib.mkBleedingPkgs {
      bleedingNixpkgs = infernisBleedingNixpkgs;
      sourcePkgs = gpuCfg.pkgs;
    };
  in {
    services.ollama =
      {
        enable = true;
        package = gpuLib.ollamaPackage {
          vendor = gpuCfg.vendor;
          pkgs = bleedingPkgs;
        };
        host = cfg.host;
        port = cfg.port;
        loadModels = cfg.loadModels;
        environmentVariables = gpuLib.deviceEnvVars {
          vendor = gpuCfg.vendor;
          inherit (gpuCfg) visibleDevices;
        };
      }
      // (
        if gpuCfg.vendor == "amd" && gpuCfg.amd.gfxVersion != null
        then {rocmOverrideGfx = gpuCfg.amd.gfxVersion;}
        else {}
      )
      // cfg.extraConfig;

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [cfg.port];
  });
}
