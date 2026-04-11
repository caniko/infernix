# Pure functions mapping GPU vendor to packages, env vars, and systemd overrides.
# No module system — importable by any module that needs GPU-aware decisions.
{lib}: let
  inherit (lib) mkForce optional mapAttrsToList concatStringsSep;

  # Environment variables for device visibility.
  deviceEnvVars = {
    vendor,
    visibleDevices,
  }: let
    devStr = concatStringsSep "," visibleDevices;
  in
    if vendor == "amd"
    then {ROCR_VISIBLE_DEVICES = devStr;}
    else if vendor == "nvidia"
    then {CUDA_VISIBLE_DEVICES = devStr;}
    else {};
in {
  inherit deviceEnvVars;

  # Re-instantiate a bleeding-edge nixpkgs flake with the same system and
  # config as the consumer's reference pkgs. This is how ollama / llama-cpp
  # / llama-swap pull from nixpkgs master while still inheriting
  # rocmSupport, cudaSupport, allowUnfree, etc. from the consumer's
  # nixos-unstable pkgs.
  mkBleedingPkgs = {
    bleedingNixpkgs,
    sourcePkgs,
  }:
    import bleedingNixpkgs {
      inherit (sourcePkgs.stdenv.hostPlatform) system;
      inherit (sourcePkgs) config;
    };

  # Select the correct ollama package based on GPU vendor.
  ollamaPackage = {
    vendor,
    pkgs,
  }:
    if vendor == "amd"
    then pkgs.ollama-rocm
    else if vendor == "nvidia"
    then pkgs.ollama-cuda
    else pkgs.ollama;

  # Systemd service overrides needed for GPU inference workloads.
  systemdGpuOverrides = {
    vendor,
    visibleDevices,
    amd ? {},
  }: let
    gfxVersion = amd.gfxVersion or null;
    envVars =
      mapAttrsToList (k: v: "${k}=${v}") (deviceEnvVars {inherit vendor visibleDevices;})
      ++ optional (vendor == "amd" && gfxVersion != null) "HSA_OVERRIDE_GFX_VERSION=${gfxVersion}";
  in
    if vendor == "amd"
    then {
      # ROCm JIT requires write+execute memory
      MemoryDenyWriteExecute = mkForce false;
      # llama-server reads /proc/meminfo for memory management
      ProcSubset = mkForce "all";
      # Pin model weights in RAM — prevent swap-out during inference
      LimitMEMLOCK = "infinity";
      # AMD GPU device access
      DevicePolicy = mkForce "auto";
      SupplementaryGroups = ["video" "render"];
      Environment = envVars;
    }
    else if vendor == "nvidia"
    then {
      DevicePolicy = mkForce "auto";
      Environment = envVars;
    }
    else {};

  # Override llama-cpp with CPU arch and GPU-specific cmake flags.
  overrideLlamaCpp = {
    vendor,
    pkgs,
    amd ? {},
    extraCmakeFlags ? [],
    flashAttention ? {},
  }: let
    cpuArch = amd.cpuArch or null;
    allQuants = flashAttention.allQuants or false;
  in
    pkgs.llama-cpp.overrideAttrs (old: {
      cmakeFlags =
        old.cmakeFlags
        ++ (optional (cpuArch != null) (lib.cmakeFeature "CMAKE_C_FLAGS" "-march=${cpuArch}"))
        ++ (optional (cpuArch != null) (lib.cmakeFeature "CMAKE_CXX_FLAGS" "-march=${cpuArch}"))
        ++ (optional (vendor == "amd" && allQuants) (lib.cmakeBool "GGML_HIP_FA_ALL_QUANTS" true))
        ++ extraCmakeFlags;
    });
}
