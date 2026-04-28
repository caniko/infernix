# Direct-wiring module for infernix → programs.goose.
#
# Import this module in addition to a Home Manager module that declares
# `programs.goose` for any user that wants infernix to auto-configure goose.
#
# This is kept separate from `homeModules.default` because `programs.goose`
# only exists when such a module is imported, and infernix's default HM module
# is typically loaded via `home-manager.sharedModules` (i.e. for every user on
# a host). Trying to write `programs.goose.*` from there would break type
# evaluation for users that don't declare `programs.goose`.
#
# Reads the readOnly outputs declared in `./goose.nix` and writes them to
# `programs.goose` when `services.infernix.goose.enable = true`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.infernix.goose;
  generatedDefaultProvider = cfg.generatedSettings.GOOSE_PROVIDER or null;
  generatedDefaultModel = cfg.generatedSettings.GOOSE_MODEL or null;
  generatedOllamaHost = cfg.generatedSettings.OLLAMA_HOST or null;
in {
  config = lib.mkIf cfg.enable {
    programs.goose = lib.mkMerge [
      {
        customProviders = cfg.generatedProviders;
      }
      (lib.mkIf (generatedDefaultProvider != null) {
        settings.GOOSE_PROVIDER = lib.mkDefault generatedDefaultProvider;
      })
      (lib.mkIf (generatedDefaultModel != null) {
        settings.GOOSE_MODEL = lib.mkDefault generatedDefaultModel;
      })
      (lib.mkIf (generatedOllamaHost != null) {
        settings.OLLAMA_HOST = lib.mkDefault generatedOllamaHost;
      })
    ];
  };
}
