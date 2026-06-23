{
  config,
  lib,
  pkgs,
  infernixManifest,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;

  cfg = config.services.infernix.manifest;
in {
  options.services.infernix.manifest = {
    enable = mkEnableOption "Manifest AI model router via infernix";

    betterAuthSecretFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to BETTER_AUTH_SECRET file. Required on first boot.";
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
    services.manifest = {
      enable = true;
      betterAuthSecretFile = cfg.betterAuthSecretFile;
      inherit (cfg) imageTag openFirewall;
    };
  };
}
