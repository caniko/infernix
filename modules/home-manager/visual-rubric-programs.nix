# Direct-wiring module for infernix → programs.visual-rubric.
#
# Import this module in addition to a Home Manager module that declares
# `programs.visual-rubric` for any user that wants infernix to auto-configure
# visual-rubric.
#
# This is kept separate from `homeModules.default` because `programs.visual-rubric`
# only exists when such a module is imported, and infernix's default HM module
# is typically loaded via `home-manager.sharedModules` (i.e. for every user on
# a host). Trying to write `programs.visual-rubric.*` from there would break type
# evaluation for users that don't declare `programs.visual-rubric`.
#
# Reads the readOnly outputs declared in `./visual-rubric.nix` and writes them
# to `programs.visual-rubric` when `services.infernix.visual-rubric.enable = true`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.infernix.visual-rubric;
  gen = cfg.generatedConfig;
in {
  config = lib.mkIf cfg.enable {
    programs.visual-rubric = lib.mkMerge [
      (lib.mkIf (gen ? vision_url) {
        settings.VISUAL_RUBRIC_VISION_URL = lib.mkDefault gen.vision_url;
      })
      (lib.mkIf (gen ? vision_model) {
        settings.VISUAL_RUBRIC_VISION_MODEL = lib.mkDefault gen.vision_model;
      })
      (lib.mkIf (gen ? rubric_backend) {
        settings.VISUAL_RUBRIC_RUBRIC_BACKEND = lib.mkDefault gen.rubric_backend;
      })
      (lib.mkIf (gen ? rubric_acp_args && builtins.length gen.rubric_acp_args > 0) {
        settings.VISUAL_RUBRIC_ACP_ARGS =
          lib.mkDefault (builtins.concatStringsSep " " gen.rubric_acp_args);
      })
    ];
  };
}
