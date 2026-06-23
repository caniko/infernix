{
  config,
  lib,
  pkgs,
  infernixMnemo,
  ...
}: let
  inherit (lib) literalExpression mkEnableOption mkIf mkOption optionals types;
  cfg = config.services.infernix.mnemo;
  system = pkgs.stdenv.hostPlatform.system;
  tomlFormat = pkgs.formats.toml {};
  surrealCfg = config.services.infernix.surrealdb;
  qdrantCfg = config.services.infernix.qdrant;

  localhostOf = host:
    if builtins.elem host [
      "0.0.0.0"
      "::"
      "[::]"
    ]
    then "127.0.0.1"
    else host;

  mkToml = password: tomlFormat.generate "mnemo-config.toml" (lib.recursiveUpdate cfg.settings {
    storage = {
      surreal_url = cfg.surrealdb.url;
      surreal_username = cfg.surrealdb.username;
      inherit password;
      surreal_namespace = cfg.surrealdb.namespace;
      surreal_database = cfg.surrealdb.database;
    };
    vector = {
      qdrant_url = cfg.qdrant.url;
      collection_name = cfg.qdrant.collectionName;
    };
  });

  mnemoConfig = mkToml cfg.surrealdb.password;

  # Template with placeholder used when passwordFile is set;
  # the placeholder is substituted at activation time so the
  # actual password never enters the Nix store.
  mnemoConfigTemplate = mkToml "@SURREALDB_PASSWORD@";
in {
  options.services.infernix.mnemo = {
    enable = mkEnableOption "Mnemo tooling and system configuration";

    package = mkOption {
      type = types.package;
      default = infernixMnemo.packages.${system}.default;
      defaultText = literalExpression "inputs.infernix.packages.<system>.mnemo";
      description = "Mnemo package bundle to install.";
    };

    settings = mkOption {
      type = tomlFormat.type;
      default = {};
      description = ''
        Additional Mnemo configuration merged into `/etc/mnemo/config.toml`.
        The module-owned SurrealDB and Qdrant connection fields override matching
        values from this attrset.
      '';
    };

    surrealdb = {
      useInfernixService = mkOption {
        type = types.bool;
        default = surrealCfg.enable;
        defaultText = literalExpression "config.services.infernix.surrealdb.enable";
        description = ''
          Whether to derive the SurrealDB endpoint from the host's
          `services.infernix.surrealdb` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernixService
          then "ws://${localhostOf surrealCfg.host}:${toString surrealCfg.port}"
          else "ws://127.0.0.1:8000";
        description = "WebSocket URL Mnemo should use to connect to SurrealDB.";
      };

      username = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernixService && surrealCfg.auth.enable
          then surrealCfg.auth.username
          else "root";
        description = "Username Mnemo should use when connecting to SurrealDB.";
      };

      password = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernixService && surrealCfg.auth.enable
          then surrealCfg.auth.password
          else "root";
        description = ''
          Password Mnemo should use when connecting to SurrealDB.

          This value is stored in the Nix store when set here. For production
          use, prefer `passwordFile` instead — it loads the password from a
          file at runtime and keeps it out of the Nix store.
        '';
      };

      passwordFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          File containing the SurrealDB password. Takes precedence over
          `password` when set. When configured, the Mnemo config file is
          generated at activation time by substituting the placeholder with
          the file contents, keeping the password out of the Nix store.

          The default for `password` is still evaluated for option display,
          but is ignored at runtime when `passwordFile` is set.
        '';
      };

      namespace = mkOption {
        type = types.str;
        default = "mnemo";
        description = "SurrealDB namespace for Mnemo data.";
      };

      database = mkOption {
        type = types.str;
        default = "mnemo";
        description = "SurrealDB database for Mnemo data.";
      };
    };

    qdrant = {
      useInfernixService = mkOption {
        type = types.bool;
        default = qdrantCfg.enable;
        defaultText = literalExpression "config.services.infernix.qdrant.enable";
        description = ''
          Whether to derive the Qdrant endpoint from the host's
          `services.infernix.qdrant` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.qdrant.useInfernixService
          then "http://${localhostOf qdrantCfg.host}:${toString qdrantCfg.grpcPort}"
          else "http://127.0.0.1:6334";
        description = "Qdrant endpoint Mnemo should use.";
      };

      collectionName = mkOption {
        type = types.str;
        default = "mnemo_embeddings";
        description = "Qdrant collection Mnemo should create or reuse.";
      };
    };
  };

  config = mkIf cfg.enable {
    assertions =
      optionals cfg.surrealdb.useInfernixService [
        {
          assertion = surrealCfg.enable;
          message = ''
            services.infernix.mnemo.surrealdb.useInfernixService = true requires
            services.infernix.surrealdb.enable = true.
          '';
        }
        {
          assertion = surrealCfg.auth.enable;
          message = ''
            services.infernix.mnemo.surrealdb.useInfernixService = true requires
            services.infernix.surrealdb.auth.enable = true because Mnemo always
            authenticates when connecting to SurrealDB.
          '';
        }
      ]
      ++ optionals cfg.qdrant.useInfernixService [
        {
          assertion = qdrantCfg.enable;
          message = ''
            services.infernix.mnemo.qdrant.useInfernixService = true requires
            services.infernix.qdrant.enable = true.
          '';
        }
      ];

    environment.systemPackages = [cfg.package];

    # When passwordFile is set, substitute the placeholder at activation
    # time so the actual password never enters the Nix store.
    system.activationScripts.mnemoConfig = mkIf (cfg.surrealdb.passwordFile != null) ''
      mkdir -p /etc/mnemo
      sed "s|@SURREALDB_PASSWORD@|$(cat "${cfg.surrealdb.passwordFile}")|g" \
        ${mnemoConfigTemplate} > /etc/mnemo/config.toml
    '';

    # Without passwordFile, use the build-time config (password stored in
    # the Nix store — the documented tradeoff in the option description).
    environment.etc."mnemo/config.toml" = mkIf (cfg.surrealdb.passwordFile == null) {
      source = mnemoConfig;
    };
  };
}
