{ config
, infernixSelf
, lib
, pkgs
, ...
}:
let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.workloadFabric;

  adapterSubmodule = types.submodule {
    options = {
      workload = mkOption {
        type = types.str;
        description = "Workload name handled by this adapter.";
      };

      queues = mkOption {
        type = types.listOf types.str;
        description = "Queues this adapter is willing to claim.";
      };

      command = mkOption {
        type = types.str;
        description = "Adapter executable. It receives one JSON job envelope on stdin.";
      };

      args = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Arguments passed to the adapter before the JSON envelope.";
      };

      workingDirectory = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Optional working directory for the adapter process.";
      };

      environment = mkOption {
        type = types.attrsOf types.str;
        default = { };
        description = "Non-secret environment variables passed to the adapter.";
      };
    };
  };

  configFile = (pkgs.formats.toml { }).generate "infernix-workerd.toml" ({
    database_url_env = cfg.databaseUrlEnv;
    worker = {
      id = cfg.workerId;
      capabilities = cfg.capabilities;
      concurrency = cfg.concurrency;
      lease_duration_secs = cfg.leaseDurationSecs;
      heartbeat_interval_secs = cfg.heartbeatIntervalSecs;
      poll_interval_secs = cfg.pollIntervalSecs;
      staging_root = toString cfg.stagingRoot;
    } // lib.optionalAttrs (cfg.bootId != null) {
      boot_id = cfg.bootId;
    };
    adapters = lib.mapAttrsToList
      (_name: adapter: {
        workload = adapter.workload;
        queues = adapter.queues;
        command = adapter.command;
        args = adapter.args;
        environment = adapter.environment;
      } // lib.optionalAttrs (adapter.workingDirectory != null) {
        working_directory = toString adapter.workingDirectory;
      })
      cfg.adapters;
  });

  workerPackage = cfg.package;
  workerExec = "${workerPackage}/bin/infernix-workerd";
  databaseEnvironmentFile = lib.optional (cfg.databaseUrlFile != null) cfg.databaseUrlFile;
  databaseEnvironment = lib.optional
    (
      cfg.databaseUrl != null
    ) "${cfg.databaseUrlEnv}=${cfg.databaseUrl}";
in
{
  options.services.infernix.workloadFabric = {
    enable = mkEnableOption "the Infernix durable workload fabric worker";

    package = mkOption {
      type = types.package;
      default = infernixSelf.packages.${pkgs.stdenv.hostPlatform.system}.infernix-workerd;
      defaultText = lib.literalExpression "inputs.infernix.packages.<system>.infernix-workerd";
      description = "Package providing the PostgreSQL-backed infernix-workerd executable.";
    };

    databaseUrlEnv = mkOption {
      type = types.str;
      default = "INFERNIX_WORKLOAD_DATABASE_URL";
      description = "Environment variable containing the PostgreSQL connection URL.";
    };

    databaseUrlFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Environment file containing databaseUrlEnv for a credential-bearing URL.";
    };

    databaseUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Non-secret PostgreSQL URL, useful for a local peer-authenticated Unix socket.";
    };

    workerId = mkOption {
      type = types.str;
      default = config.networking.hostName;
      defaultText = lib.literalExpression "config.networking.hostName";
      description = "Stable worker identity. The boot ID fences restarts of this worker.";
    };

    bootId = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Optional fixed boot identity; otherwise the daemon reads the kernel boot ID.";
    };

    capabilities = mkOption {
      type = types.listOf types.str;
      default = [ "cpu" ];
      description = "Capability labels matched against queued job requirements.";
    };

    concurrency = mkOption {
      type = types.ints.positive;
      default = 1;
      description = "Maximum number of concurrently leased jobs on this worker.";
    };

    leaseDurationSecs = mkOption {
      type = types.ints.positive;
      default = 900;
      description = "Lease duration; must exceed heartbeatIntervalSecs.";
    };

    heartbeatIntervalSecs = mkOption {
      type = types.ints.positive;
      default = 30;
      description = "How often a running adapter renews its lease.";
    };

    pollIntervalSecs = mkOption {
      type = types.ints.positive;
      default = 5;
      description = "Delay between empty claim polls.";
    };

    stagingRoot = mkOption {
      type = types.path;
      default = "/var/lib/infernix-workload/staging";
      description = "Local immutable-artifact staging root for adapter attempts.";
    };

    adapters = mkOption {
      type = types.attrsOf adapterSubmodule;
      default = { };
      description = "Workload adapters available to this worker.";
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.databaseUrlFile != null || cfg.databaseUrl != null;
        message = "services.infernix.workloadFabric needs databaseUrlFile or a non-secret databaseUrl when enabled.";
      }
      {
        assertion = cfg.heartbeatIntervalSecs < cfg.leaseDurationSecs;
        message = "services.infernix.workloadFabric.heartbeatIntervalSecs must be less than leaseDurationSecs.";
      }
    ]
    ++ lib.mapAttrsToList
      (name: adapter: {
        assertion = adapter.queues != [ ];
        message = "services.infernix.workloadFabric.adapters.${name}.queues must not be empty.";
      })
      cfg.adapters;

    systemd.services.infernix-workload-migrate = {
      description = "Prepare the Infernix durable workload schema";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${workerExec} --config ${configFile} migrate";
        EnvironmentFile = databaseEnvironmentFile;
        Environment = databaseEnvironment;
        RemainAfterExit = true;
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      };
    };

    systemd.services.infernix-workerd = {
      description = "Infernix durable workload worker";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" "infernix-workload-migrate.service" ];
      wants = [ "network-online.target" ];
      requires = [ "infernix-workload-migrate.service" ];
      serviceConfig = {
        ExecStart = "${workerExec} --config ${configFile} worker";
        EnvironmentFile = databaseEnvironmentFile;
        Environment = databaseEnvironment;
        StateDirectory = "infernix-workload";
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      };
    };
  };
}
