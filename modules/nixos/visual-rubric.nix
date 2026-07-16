{
  config,
  lib,
  pkgs,
  infernixVisualRubric,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types;
  cfg = config.services.infernix.visual-rubric;
  toml = pkgs.formats.toml {};
  generatedConfig = toml.generate "visual-rubric-config.toml" cfg.settings;
in {
  options.services.infernix.visual-rubric = {
    enable = mkEnableOption "visual-rubric package and system integration";

    package = mkOption {
      type = types.nullOr types.package;
      default =
        if infernixVisualRubric ? packages.${pkgs.stdenv.hostPlatform.system}.default
        then infernixVisualRubric.packages.${pkgs.stdenv.hostPlatform.system}.default
        else null;
      defaultText = lib.literalExpression "infernixVisualRubric.packages.\${system}.default";
      description = "The visual-rubric package to use.";
    };

    settings = mkOption {
      type = toml.type;
      default = {};
      description = ''
        visual-rubric TOML configuration. This is rendered once by the
        NixOS module so system services do not hand-write a second backend
        configuration.
      '';
    };

    configFile = mkOption {
      type = types.path;
      readOnly = true;
      default = generatedConfig;
      description = "Generated visual-rubric configuration file.";
    };
  };

  config = mkIf cfg.enable {
    environment.systemPackages = lib.optional (cfg.package != null) cfg.package;
    environment.etc."visual-rubric/config.toml".source = cfg.configFile;
    assertions = [
      {
        assertion = cfg.package != null;
        message = ''
          services.infernix.visual-rubric is enabled but no package could be
          resolved; set services.infernix.visual-rubric.package explicitly.
        '';
      }
    ];
  };
}
