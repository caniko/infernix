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

  # Re-instantiate the locked nixos-unstable nixpkgs flake with the same
  # system and config as the consumer's reference pkgs. This lets GPU-aware
  # packages inherit rocmSupport, cudaSupport, allowUnfree, etc.
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

  # Override llama-cpp with GPU-specific cmake flags and an optional
  # CPU-microarch hardware profile.
  #
  # `hardwareOptimization` accepts a crossbow-shaped profile attrset with
  # `.platform.gcc.arch` (and optionally `.platform.gcc.tune`), or `null`
  # to keep the cached upstream binary untouched. The flake input that
  # provides the profile (e.g. nix-crossbow) is the consumer's concern —
  # infernix only consumes the resolved attrset to avoid a hard dep.
  overrideLlamaCpp = {
    vendor,
    pkgs,
    hardwareOptimization ? null,
    extraCmakeFlags ? [],
    flashAttention ? {},
  }: let
    allQuants = flashAttention.allQuants or false;
    gcc =
      if hardwareOptimization == null
      then null
      else hardwareOptimization.platform.gcc or null;
    archFlags =
      if gcc == null
      then []
      else
        optional (gcc ? arch) "-march=${gcc.arch}"
        ++ optional (gcc ? tune) "-mtune=${gcc.tune}";
    # Skip the override entirely when nothing would change — preserves the
    # binary cache hit for hosts that don't opt into a rebuild.
    needsOverride =
      archFlags != []
      || extraCmakeFlags != []
      || (vendor == "amd" && allQuants);
  in
    if !needsOverride
    then pkgs.llama-cpp
    else
      pkgs.llama-cpp.overrideAttrs (old: {
        cmakeFlags =
          old.cmakeFlags
          ++ (optional (archFlags != []) (lib.cmakeFeature "CMAKE_C_FLAGS" (concatStringsSep " " archFlags)))
          ++ (optional (archFlags != []) (lib.cmakeFeature "CMAKE_CXX_FLAGS" (concatStringsSep " " archFlags)))
          ++ (optional (vendor == "amd" && allQuants) (lib.cmakeBool "GGML_HIP_FA_ALL_QUANTS" true))
          ++ extraCmakeFlags;
      });
}
