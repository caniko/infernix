{
  config,
  lib,
  pkgs,
  ...
}:
let
  inherit (lib)
    literalExpression
    mkDefault
    mkEnableOption
    mkIf
    mkOption
    optionals
    types
    ;
  cfg = config.services.infernix.surrealdb;

  backendToDbPath =
    backend:
    if backend == "rocksdb" then
      "rocksdb:///var/lib/surrealdb/"
    else if backend == "memory" then
      "memory"
    else
      "surrealkv:///var/lib/surrealdb/";

  # Local proof-layer variant that matches the planned nixpkgs default:
  # SurrealKV + memory without the RocksDB env wiring. bindgenHook stays
  # because rquickjs-sys still needs libclang during the build.
  surrealdbSurrealKv = pkgs.surrealdb.overrideAttrs (old: {
    buildNoDefaultFeatures = true;
    buildFeatures = [
      "allocator"
      "allocation-tracking"
      "http"
      "scripting"
      "storage-mem"
      "storage-surrealcs"
      "storage-surrealkv"
    ];

    env = builtins.removeAttrs (old.env or { }) [
      "ROCKSDB_INCLUDE_DIR"
      "ROCKSDB_LIB_DIR"
    ];

  });

  defaultPackage = if cfg.backend == "rocksdb" then pkgs.surrealdb else surrealdbSurrealKv;

  effectiveDbPath = if cfg.dbPath != null then cfg.dbPath else backendToDbPath cfg.backend;
in
{
  options.services.infernix.surrealdb = {
    enable = mkEnableOption "SurrealDB multi-model database";

    package = lib.mkPackageOption pkgs "surrealdb" { };

    backend = mkOption {
      type = types.enum [
        "surrealkv"
        "rocksdb"
        "memory"
      ];
      default = "surrealkv";
      description = ''
        Storage backend infernix should target when `dbPath` is not set.
        `surrealkv` is the default, `rocksdb` uses the upstream package as-is,
        and `memory` runs without on-disk persistence.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Address to bind SurrealDB to.";
    };

    port = mkOption {
      type = types.port;
      default = 8000;
      description = "SurrealDB HTTP API port.";
    };

    dbPath = mkOption {
      type = types.nullOr types.str;
      default = null;
      defaultText = literalExpression ''
        if config.services.infernix.surrealdb.backend == "rocksdb"
        then "rocksdb:///var/lib/surrealdb/"
        else if config.services.infernix.surrealdb.backend == "memory"
        then "memory"
        else "surrealkv:///var/lib/surrealdb/"
      '';
      example = "memory";
      description = ''
        Raw storage backend URI passed to `surreal start`. If unset, infernix
        derives it from `backend`. The packaged defaults cover
        `surrealkv:///...`, `rocksdb:///...`, and `memory`; other URI schemes
        require a custom `package` that enables the corresponding backend.
      '';
    };

    extraFlags = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = [
        "--allow-all"
        "--auth"
        "--user"
        "root"
        "--pass"
        "root"
      ];
      description = ''
        Additional CLI flags appended to `surreal start` after infernix-managed
        auth flags.
      '';
    };

    auth = {
      enable = mkEnableOption "SurrealDB root authentication";

      username = mkOption {
        type = types.str;
        default = "root";
        description = "Root username passed to `surreal start` when auth is enabled.";
      };

      password = mkOption {
        type = types.str;
        default = "root";
        description = ''
          Root password passed to `surreal start` when auth is enabled.
          This value is stored in the Nix store; prefer a localhost bind if you
          keep the default and do not expose the port publicly.
        '';
      };
    };

    openFirewall = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to open firewall ports for SurrealDB.";
    };
  };

  config = mkIf cfg.enable {
    services.infernix.surrealdb.package = mkDefault defaultPackage;

    services.surrealdb = {
      enable = true;
      package = cfg.package;
      host = cfg.host;
      port = cfg.port;
      dbPath = effectiveDbPath;
      extraFlags =
        optionals cfg.auth.enable [
          "--auth"
          "--user"
          cfg.auth.username
          "--pass"
          cfg.auth.password
        ]
        ++ cfg.extraFlags;
    };

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];
  };
}
