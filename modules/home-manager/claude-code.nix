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
  # KEY=VALUE file (mode 0600) shared by the provider, the router and shells.
  codexTokenFile = "${config.xdg.stateHome}/infernix/codex-provider.env";
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
    # The router forwards to the Codex provider, so it needs the same secret
    # rather than a world-readable store value. CCR expands $VARS itself.
    APIKEY =
      if cfg.apiKey == null
      then "$INFERNIX_CODEX_PROVIDER_TOKEN"
      else cfg.apiKey;
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
      type = types.nullOr types.str;
      default = null;
      description = ''
        Local bearer token shared by Claude Code and the router. Null uses the
        per-user token generated for the Codex provider; a string is stored in
        the world-readable Nix store.
      '';
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
      ANTHROPIC_BASE_URL = lib.mkDefault "http://127.0.0.1:${toString cfg.routerPort}";
      ANTHROPIC_AUTH_TOKEN = lib.mkIf (cfg.apiKey != null) cfg.apiKey;
      CODEX_HOME = "${config.home.homeDirectory}/.codex";
    };

    home.activation.infernixCodexProviderToken = lib.hm.dag.entryAfter ["writeBoundary"] ''
      if [ ! -s ${lib.escapeShellArg codexTokenFile} ]; then
        install -d -m 0700 "$(dirname ${lib.escapeShellArg codexTokenFile})"
        (umask 077; printf 'INFERNIX_CODEX_PROVIDER_TOKEN=%s\n' "$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')" > ${lib.escapeShellArg codexTokenFile})
      fi
    '';

    home.sessionVariablesExtra = ''
      if [ -r ${lib.escapeShellArg codexTokenFile} ]; then
        set -a
        . ${lib.escapeShellArg codexTokenFile}
        set +a
        ${lib.optionalString (cfg.apiKey == null) ''export ANTHROPIC_AUTH_TOKEN="$INFERNIX_CODEX_PROVIDER_TOKEN"''}
      fi
    '';

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
        EnvironmentFile = codexTokenFile;
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
        EnvironmentFile = codexTokenFile;
        Restart = "on-failure";
        Environment = ["HOME=${config.home.homeDirectory}"];
      };
      Install.WantedBy = ["default.target"];
    };
  };
}
