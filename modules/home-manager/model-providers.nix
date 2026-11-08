{
  config,
  lib,
  ...
}: let
  inherit (lib) mkOption types;
  defaults = import ../../lib/model-providers.nix {inherit lib;};
  defaultProviders = defaults.providers // {
    codex = defaults.providers.codex // {
      baseUrl = "http://127.0.0.1:${toString config.services.infernix.modelProviders.codexProviderPort}/v1";
    };
  };
in {
  options.services.infernix.modelProviders = {
    providers = mkOption {
      type = types.attrs;
      default = defaultProviders;
      description = "Shared provider and model catalog rendered for agent clients.";
    };

    routes = mkOption {
      type = types.attrs;
      default = defaults.routes;
      description = "Shared model routing defaults.";
    };

    codexProviderPort = mkOption {
      type = types.port;
      default = defaults.codexProviderPort;
      description = "Loopback port for the shared Codex CLI provider.";
    };
  };

  config.services.infernix.modelProviders = {
    providers = lib.mkDefault defaultProviders;
    routes = lib.mkDefault defaults.routes;
    codexProviderPort = lib.mkDefault defaults.codexProviderPort;
  };
}
