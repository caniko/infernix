{
  config,
  lib,
  pkgs,
  infernisMnemo,
  ...
}: let
  inherit (lib) literalExpression mkEnableOption mkIf mkOption optionals types;
  cfg = config.services.infernis.mnemo;
  system = pkgs.stdenv.hostPlatform.system;
  tomlFormat = pkgs.formats.toml {};
  surrealCfg = config.services.infernis.surrealdb;
  qdrantCfg = config.services.infernis.qdrant;

  localhostOf = host:
    if builtins.elem host [
      "0.0.0.0"
      "::"
      "[::]"
    ]
    then "127.0.0.1"
    else host;

  mnemoConfig = tomlFormat.generate "mnemo-config.toml" (lib.recursiveUpdate cfg.settings {
    storage = {
      surreal_url = cfg.surrealdb.url;
      surreal_username = cfg.surrealdb.username;
      surreal_password = cfg.surrealdb.password;
      surreal_namespace = cfg.surrealdb.namespace;
      surreal_database = cfg.surrealdb.database;
    };
    vector = {
      qdrant_url = cfg.qdrant.url;
      collection_name = cfg.qdrant.collectionName;
    };
  });
in {
  options.services.infernis.mnemo = {
    enable = mkEnableOption "Mnemo tooling and system configuration";

    package = mkOption {
      type = types.package;
      default = infernisMnemo.packages.${system}.default;
      defaultText = literalExpression "inputs.infernis.packages.<system>.mnemo";
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
      useInfernisService = mkOption {
        type = types.bool;
        default = surrealCfg.enable;
        defaultText = literalExpression "config.services.infernis.surrealdb.enable";
        description = ''
          Whether to derive the SurrealDB endpoint from the host's
          `services.infernis.surrealdb` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernisService
          then "ws://${localhostOf surrealCfg.host}:${toString surrealCfg.port}"
          else "ws://127.0.0.1:8000";
        description = "WebSocket URL Mnemo should use to connect to SurrealDB.";
      };

      username = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernisService && surrealCfg.auth.enable
          then surrealCfg.auth.username
          else "root";
        description = "Username Mnemo should use when connecting to SurrealDB.";
      };

      password = mkOption {
        type = types.str;
        default =
          if cfg.surrealdb.useInfernisService && surrealCfg.auth.enable
          then surrealCfg.auth.password
          else "root";
        description = "Password Mnemo should use when connecting to SurrealDB.";
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
      useInfernisService = mkOption {
        type = types.bool;
        default = qdrantCfg.enable;
        defaultText = literalExpression "config.services.infernis.qdrant.enable";
        description = ''
          Whether to derive the Qdrant endpoint from the host's
          `services.infernis.qdrant` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.qdrant.useInfernisService
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
      optionals cfg.surrealdb.useInfernisService [
        {
          assertion = surrealCfg.enable;
          message = ''
            services.infernis.mnemo.surrealdb.useInfernisService = true requires
            services.infernis.surrealdb.enable = true.
          '';
        }
        {
          assertion = surrealCfg.auth.enable;
          message = ''
            services.infernis.mnemo.surrealdb.useInfernisService = true requires
            services.infernis.surrealdb.auth.enable = true because Mnemo always
            authenticates when connecting to SurrealDB.
          '';
        }
      ]
      ++ optionals cfg.qdrant.useInfernisService [
        {
          assertion = qdrantCfg.enable;
          message = ''
            services.infernis.mnemo.qdrant.useInfernisService = true requires
            services.infernis.qdrant.enable = true.
          '';
        }
      ];

    environment.systemPackages = [cfg.package];
    environment.etc."mnemo/config.toml".source = mnemoConfig;
  };
}
