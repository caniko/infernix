{
  config,
  lib,
  pkgs,
  infernixManifest,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;

  cfg = config.services.infernix.manifest;
  manifestModule = infernixManifest.nixosModules.default;
in {
  options.services.infernix.manifest = {
    enable = mkEnableOption "Manifest AI model router via infernix";

    betterAuthSecretFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to BETTER_AUTH_SECRET file. Required on first boot.";
    };

    apiKeyEnvVars = mkOption {
      type = types.attrsOf types.path;
      default = {};
      description = "Env var names → file paths for cloud provider API keys. Written to Manifest environment.";
    };

    imageTag = mkOption {
      type = types.str;
      default = "latest";
      description = "Manifest Docker image tag.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open firewall for Manifest port.";
    };
  };

  config = mkIf cfg.enable {
    imports = [manifestModule];

    services.manifest = {
      enable = true;
      betterAuthSecretFile = cfg.betterAuthSecretFile;
      inherit (cfg) imageTag openFirewall;
    };
  };
}
