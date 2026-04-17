# Direct-wiring module for infernis → programs.goose.
#
# Import this module in addition to a Home Manager module that declares
# `programs.goose` for any user that wants infernis to auto-configure goose.
#
# This is kept separate from `homeModules.default` because `programs.goose`
# only exists when such a module is imported, and infernis's default HM module
# is typically loaded via `home-manager.sharedModules` (i.e. for every user on
# a host). Trying to write `programs.goose.*` from there would break type
# evaluation for users that don't declare `programs.goose`.
#
# Reads the readOnly outputs declared in `./goose.nix` and writes them to
# `programs.goose` when `services.infernis.goose.enable = true`.
{
  config,
  lib,
  ...
}: let
  cfg = config.services.infernis.goose;
  generatedDefaultProvider = cfg.generatedSettings.GOOSE_PROVIDER or null;
  generatedDefaultModel = cfg.generatedSettings.GOOSE_MODEL or null;
  generatedOllamaHost = cfg.generatedSettings.OLLAMA_HOST or null;
in {
  config = lib.mkIf cfg.enable {
    programs.goose = lib.mkMerge [
      {
        enable = true;
        customProviders = cfg.generatedProviders;
      }
      (lib.mkIf (generatedDefaultProvider != null) {
        defaultProvider = lib.mkDefault generatedDefaultProvider;
      })
      (lib.mkIf (generatedDefaultModel != null) {
        defaultModel = lib.mkDefault generatedDefaultModel;
      })
      (lib.mkIf (generatedOllamaHost != null) {
        providers.ollama.settings.OLLAMA_HOST = lib.mkDefault generatedOllamaHost;
      })
    ];
  };
}
