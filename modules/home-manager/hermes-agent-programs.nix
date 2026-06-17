{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkIf mkOption types;
  cfg = config.services.infernix.hermes-agent;
  upCfg = config.services.hermes-agent or null;
in {
  options = {
    # Expose the same option tree so the companion is transparent.
    services.infernix.hermes-agent = mkOption {
      type = types.submoduleWith {
        modules = [./hermes-agent.nix];
      };
    };
  };

  config = mkIf (cfg.enable && cfg.home.enable) {
    home.packages = mkIf cfg.home.enablePackage (
      let
        pkg =
          if cfg.home.package != null
          then cfg.home.package
          else if upCfg != null && upCfg.enable
          then upCfg.package
          else null;
      in
        lib.optional (pkg != null) pkg
    );

    home.sessionVariables =
      {
        HERMES_HOME =
          if upCfg != null && upCfg.enable
          then "${upCfg.stateDir}/.hermes"
          else "${config.home.homeDirectory}/.hermes";
      }
      // lib.optionalAttrs (cfg.generatedProviders != {}) {
        INFERNIX_HERMES_PROVIDERS = builtins.toJSON cfg.generatedProviders;
      };
  };
}
