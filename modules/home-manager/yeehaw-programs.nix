# Direct-wiring module for infernis → programs.yh.
#
# Import this module in addition to a Home Manager module that declares
# `programs.yh` for any user that wants infernis to auto-configure yeeHaw
# steeds.
#
# This is kept separate from `homeModules.default` because `programs.yh`
# only exists when yeeHaw's HM module is imported, and infernis's default HM
# module is typically loaded via `home-manager.sharedModules` (i.e. for every
# user on a host). Trying to write `programs.yh.*` from there would break type
# evaluation for users that don't declare `programs.yh`.
#
# Reads the readOnly outputs declared in `./yeehaw.nix` and writes them to
# `programs.yh.steeds` when `services.infernis.yeehaw.enable = true`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.infernis.yeehaw;
in {
  config = lib.mkIf cfg.enable {
    programs.yh.steeds = cfg.generatedSteeds;
  };
}
