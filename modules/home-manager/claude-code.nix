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
  keyFile = config.services.infernix.modelProviders.codexProviderKeyFile;
  codexProvider = providers.codex;
  claude = cfg.package;
  codexBridge = cfg.codexProviderPackage;
  loadCredential = import ../../lib/codex-credentials.nix {
    inherit lib keyFile;
    package = codexBridge;
  };
  # Dedicated state avoids reusing old router databases with public local keys.
  routerHome = "${builtins.dirOf keyFile}/claude-router";
  routerWrapper = pkgs.writeShellApplication {
    name = "infernix-claude-router";
    text = ''
      ${loadCredential}
      export HOME=${lib.escapeShellArg routerHome}
      export XDG_CONFIG_HOME="$HOME/.config"
      export XDG_DATA_HOME="$HOME/.local/share"
      # The pinned CCR 2.0.0 start command runs the listener in this process.
      exec ${cfg.routerPackage}/bin/ccr start
    '';
  };
  claudeWrapper = pkgs.writeShellApplication {
    name = "claude";
    text = ''
      ${loadCredential}
      export ANTHROPIC_AUTH_TOKEN="$INFERNIX_CODEX_PROVIDER_API_KEY"
      headers="X-Infernix-Cwd: ''${PWD}"
      if [ -n "''${ANTHROPIC_CUSTOM_HEADERS:-}" ]; then
        headers="''${ANTHROPIC_CUSTOM_HEADERS}"$'\n'"''${headers}"
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
    APIKEY = "$INFERNIX_CODEX_PROVIDER_API_KEY";
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

    services.infernix.modelProviders = {
      codexProviderEnable = true;
      codexProviderPackage = lib.mkDefault codexBridge;
    };

    home.sessionVariables = {
      ANTHROPIC_BASE_URL = "http://127.0.0.1:${toString cfg.routerPort}";
      CODEX_HOME = "${config.home.homeDirectory}/.codex";
    };

    home.file.infernix-claude-router-config = {
      target = "${routerHome}/.claude-code-router/config.json";
      force = true;
      text = builtins.toJSON routerConfig;
    };

    systemd.user.services.claude-code-router = {
      Unit = {
        Description = "Claude Code Router for Infernix providers";
        Requires = ["infernix-codex-provider.service"];
        After = ["infernix-codex-provider.service"];
      };
      Service = {
        ExecStart = lib.getExe routerWrapper;
        Restart = "on-failure";
        UMask = "0077";
      };
      Install.WantedBy = ["default.target"];
    };
  };
}
