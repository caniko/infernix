{lib, ...}: let
  inherit (lib) mkOption types;
in {
  options.services.infernis.gpu = {
    vendor = mkOption {
      type = types.enum ["amd" "nvidia" "cpu"];
      description = "GPU vendor for inference acceleration.";
    };

    visibleDevices = mkOption {
      type = types.listOf types.str;
      default = ["0"];
      description = "Device indices to expose to inference runtimes.";
    };

    pkgs = mkOption {
      type = types.unspecified;
      description = ''
        Package set with GPU support enabled.
        Pass pkgsPrimaryGpu from canix or any nixpkgs instantiation with
        rocmSupport/cudaSupport set appropriately.
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
}
