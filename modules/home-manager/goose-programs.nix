# Direct-wiring module for infernis → goose-hm.
#
# Import this module in addition to `goose-hm`'s HM module for any user that
# wants infernis to auto-configure goose:
#
#   imports = [
#     inputs.goose-hm.homeManagerModules.default
#     inputs.infernis.homeModules.goose
#   ];
#
# This is kept separate from `homeModules.default` because `programs.goose`
# only exists when goose-hm is imported, and infernis's default HM module is
# typically loaded via `home-manager.sharedModules` (i.e. for every user on a
# host). Trying to write `programs.goose.*` from there would break type
# evaluation for users that don't import goose-hm.
#
# Reads the readOnly outputs declared in `./goose.nix` and writes them to
# `programs.goose` when `services.infernis.goose.enable = true`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.infernis.goose;
in {
  config = lib.mkIf cfg.enable {
    programs.goose = {
      enable = true;
      customProviders = cfg.generatedProviders;
      settings = cfg.generatedSettings;
    };
  };
}
