{
  config,
  lib,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix;
  needsGpu =
    (cfg.ollama.enable or false)
    || (cfg.llama-swap.enable or false);
in {
  options.services.infernix.gpu = {
    vendor = mkOption {
      type = types.nullOr (types.enum ["amd" "nvidia" "cpu"]);
      default = null;
      description = ''
        GPU vendor for inference acceleration. Required when
        `services.infernix.ollama.enable` or
        `services.infernix.llama-swap.enable` is true. Left null on hosts
        that import infernix but don't enable any GPU-consuming service.
      '';
    };

    visibleDevices = mkOption {
      type = types.listOf types.str;
      default = ["0"];
      description = "Device indices to expose to inference runtimes.";
    };

    pkgs = mkOption {
      type = types.nullOr types.unspecified;
      default = null;
      description = ''
        Package set with GPU support enabled.
        Pass a nixpkgs instantiation with rocmSupport/cudaSupport set
        appropriately for the configured vendor. Required when an
        infernix service that needs the GPU is enabled.
      '';
    };

    amd = {
      gfxVersion = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "11.0.0";
        description = "HSA_OVERRIDE_GFX_VERSION value (e.g. 11.0.0 for gfx1100).";
      };

      cpuArch = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "znver4";
        description = "CPU microarchitecture for -march flag on llama-cpp.";
      };
    };
  };

  config = lib.mkIf needsGpu {
    assertions = [
      {
        assertion = cfg.gpu.vendor != null;
        message = ''
          services.infernix.gpu.vendor must be set when
          services.infernix.ollama.enable or
          services.infernix.llama-swap.enable is true.
        '';
      }
      {
        assertion = cfg.gpu.pkgs != null;
        message = ''
          services.infernix.gpu.pkgs must be set when
          services.infernix.ollama.enable or
          services.infernix.llama-swap.enable is true.
        '';
      }
    ];
  };
}
