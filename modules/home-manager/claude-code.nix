{
  config,
  infernixCodexProvider ? null,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.claude-code;
  providers = config.services.infernix.modelProviders.providers;
  routes = config.services.infernix.modelProviders.routes;
  codexProviderPort = config.services.infernix.modelProviders.codexProviderPort;
  codexProvider = providers.codex;
  claude = cfg.package;
  codexBridge = cfg.codexProviderPackage;
  claudeWrapper = pkgs.writeShellApplication {
    name = "claude";
    text = ''
      headers="X-Infernix-Cwd: ''${PWD}"
      if [ -n "''${ANTHROPIC_CUSTOM_HEADERS:-}" ]; then
        headers="''${ANTHROPIC_CUSTOM_HEADERS}, ''${headers}"
      fi
      export ANTHROPIC_CUSTOM_HEADERS="$headers"
      exec ${lib.getExe claude} "$@"
    '';
  };

  ccrProvider = _name: value: {
    name = _name;
    api_base_url = "${value.baseUrl}/chat/completions";
    api_key =
      if value ? apiKeyEnv
      then "$" + value.apiKeyEnv
      else value.apiKey;
    models = builtins.attrNames value.models;
  };

  routerConfig = {
    APIKEY = cfg.apiKey;
    HOST = "127.0.0.1";
    PORT = cfg.routerPort;
    LOG = false;
    NON_INTERACTIVE_MODE = true;
    API_TIMEOUT_MS = 600000;
    Providers = lib.mapAttrsToList ccrProvider providers;
    Router = routes;
  };
in {
  options.services.infernix.claude-code = {
    enable = mkEnableOption "Claude Code through the Infernix model gateway";

    package = mkOption {
      type = types.package;
      default = pkgs.claude-code;
      description = "Claude Code CLI package.";
    };

    routerPackage = mkOption {
      type = types.package;
      default = pkgs.claude-code-router;
      description = "Claude Code Router package.";
    };

    codexProviderPackage = mkOption {
      type = types.nullOr types.package;
      default = infernixCodexProvider;
      description = "Infernix's direct Codex CLI HTTP provider package.";
    };

    routerPort = mkOption {
      type = types.port;
      default = 3456;
      description = "Loopback port for Claude Code Router.";
    };

    apiKey = mkOption {
      type = types.str;
      default = "infernix-local";
      description = "Local bearer token shared by Claude Code and the router.";
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = codexBridge != null;
        message = "services.infernix.claude-code requires infernixCodexProvider.";
      }
      {
        assertion = providers ? codex && codexProvider.baseUrl != null;
        message = "services.infernix.claude-code requires a codex model provider.";
      }
    ];

    home.packages = [claudeWrapper cfg.routerPackage codexBridge];

    home.sessionVariables = {
      ANTHROPIC_BASE_URL = "http://127.0.0.1:${toString cfg.routerPort}";
      ANTHROPIC_AUTH_TOKEN = cfg.apiKey;
      CODEX_HOME = "${config.home.homeDirectory}/.codex";
    };

    xdg.configFile."claude-code-router/config.json" = {
      force = true;
      text = builtins.toJSON routerConfig;
    };

    systemd.user.services.infernix-codex-provider = {
      Unit = {
        Description = "Infernix direct Codex CLI provider";
      };
      Service = {
        ExecStart = "${codexBridge}/bin/infernix-codex-provider";
        Restart = "on-failure";
        Environment = [
          "CODEX_HOME=${config.home.homeDirectory}/.codex"
          "CODEX_PATH=${lib.getExe pkgs.codex}"
          "INFERNIX_CODEX_PROVIDER_PORT=${toString codexProviderPort}"
          "INFERNIX_CODEX_MODELS=${builtins.concatStringsSep "," (builtins.attrNames codexProvider.models)}"
        ];
      };
      Install.WantedBy = ["default.target"];
    };

    systemd.user.services.claude-code-router = {
      Unit = {
        Description = "Claude Code Router for Infernix providers";
        Requires = ["infernix-codex-provider.service"];
        After = ["infernix-codex-provider.service"];
      };
      Service = {
        ExecStart = "${cfg.routerPackage}/bin/ccr start";
        Restart = "on-failure";
        Environment = ["HOME=${config.home.homeDirectory}"];
      };
      Install.WantedBy = ["default.target"];
    };
  };
}
