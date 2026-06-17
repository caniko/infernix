{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.hermes-agent;
  endpoints = config.services.infernix.endpoints;

  # Discover model providers from infernix endpoints.  Each registered
  # endpoint whose type is "llama-swap" or "ollama" becomes a candidate
  # hermes provider when the endpoint carries chat-capable models.
  generatedProviders = let
    isChatEndpoint = name: ep:
      ep.type == "llama-swap" || ep.type == "ollama";
    chatEndpoints = lib.filterAttrs isChatEndpoint endpoints;
  in
    lib.mapAttrsToList (_name: ep: {
      provider = ep.type;
      baseUrl = ep.url;
    })
    chatEndpoints;
in {
  options.services.infernix.hermes-agent = {
    home = {
      enable = mkEnableOption "Hermes CLI integration in home-manager";

      enablePackage = mkOption {
        type = types.bool;
        default = true;
        description = "Add hermes CLI to home.packages.";
      };

      package = mkOption {
        type = types.nullOr types.package;
        default = null;
        description = ''
          hermes-agent package for the CLI. Defaults to
          config.services.infernix.hermes-agent.package
          (only available when NixOS-integrated).
        '';
      };
    };

    generatedProviders = mkOption {
      type = types.attrsOf (types.submodule {
        options = {
          provider = mkOption {
            type = types.str;
            description = "Endpoint type (llama-swap or ollama).";
          };
          baseUrl = mkOption {
            type = types.str;
            description = "Endpoint base URL for the model provider.";
          };
        };
      });
      readOnly = true;
      default = {};
      description = ''
        Auto-generated model provider config from infernix endpoints.
        Populated when infernix endpoints are declared.
      '';
    };
  };

  config.services.infernix.hermes-agent.generatedProviders =
    lib.listToAttrs (map (ep: lib.nameValuePair ep.provider ep) generatedProviders);
}
