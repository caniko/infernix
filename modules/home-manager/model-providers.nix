{
  config,
  infernixCodexProvider ? null,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.modelProviders;
  defaults = import ../../lib/model-providers.nix {inherit lib;};
  defaultProviders =
    defaults.providers
    // {
      codex =
        defaults.providers.codex
        // {
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

    codexProviderEnable = lib.mkEnableOption "the private per-user Codex CLI provider";

    codexProviderPackage = mkOption {
      type = types.nullOr types.package;
      default = infernixCodexProvider;
      description = "Codex HTTP provider and private credential helper package.";
    };

    codexProviderKeyFile = mkOption {
      type = types.str;
      default = "${config.xdg.stateHome}/infernix/codex-provider.key";
      description = "Private runtime credential file (0600). Activation creates a persistent random key; its value never enters the Nix store.";
    };
  };

  config = lib.mkMerge [
    {
      services.infernix.modelProviders = {
        providers = lib.mkDefault defaultProviders;
        routes = lib.mkDefault defaults.routes;
        codexProviderPort = lib.mkDefault defaults.codexProviderPort;
      };
    }
    (lib.mkIf cfg.codexProviderEnable {
      assertions = [
        {
          assertion = cfg.codexProviderPackage != null && cfg.providers ? codex;
          message = "The Codex provider requires its package and model catalog.";
        }
      ];
      home.activation.infernixCodexCredential = lib.hm.dag.entryBetween ["linkGeneration"] ["writeBoundary"] ''
        run ${cfg.codexProviderPackage}/bin/infernix-codex-credentials ensure ${lib.escapeShellArg cfg.codexProviderKeyFile}
      '';
      systemd.user.services.infernix-codex-provider = {
        Unit.Description = "Infernix authenticated Codex CLI provider";
        Service = {
          ExecStart = "${cfg.codexProviderPackage}/bin/infernix-codex-provider";
          Restart = "on-failure";
          UMask = "0077";
          Environment = [
            "CODEX_HOME=${config.home.homeDirectory}/.codex"
            "CODEX_PATH=${lib.getExe pkgs.codex}"
            "INFERNIX_CODEX_PROVIDER_PORT=${toString cfg.codexProviderPort}"
            "INFERNIX_CODEX_PROVIDER_KEY_FILE=${cfg.codexProviderKeyFile}"
            "INFERNIX_CODEX_MODELS=${builtins.concatStringsSep "," (builtins.attrNames cfg.providers.codex.models)}"
          ];
        };
        Install.WantedBy = ["default.target"];
      };
    })
  ];
}
