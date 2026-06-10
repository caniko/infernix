{
  config,
  lib,
  pkgs,
  infernixVisualRubric,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types;
  cfg = config.services.infernix.visual-rubric;
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
  };

  config = mkIf cfg.enable {
    environment.systemPackages = lib.optional (cfg.package != null) cfg.package;
  };
}
