{ config
, infernixSelf
, lib
, pkgs
, ...
}:
let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.workloadFabric;
  gpuGateCfg = config.services.infernix.gpuGate;
  gpuGate = import ../../lib/gpu-gate.nix {inherit lib pkgs;};
  workerUser = "infernix-worker";

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

      profiles = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = ''
          Model profile ids this adapter serves. A job's `profile:<id>`
          requirement selects the adapter, and the worker advertises one
          `profile:<id>` capability per entry.
        '';
      };

      artifacts = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Validate the adapter's result manifest and deliver its artifacts.
          Set for adapters that print envelope-v2 artifacts.
        '';
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

  admissionSubmodule = types.submodule {
    options = {
      command = mkOption {
        type = types.listOf types.str;
        description = ''
          Host admission probe. Exit 0 admits GPU work; any other exit code, a
          signal, a timeout or a spawn failure denies (fail closed).
        '';
      };

      recheckSecs = mkOption {
        type = types.ints.positive;
        default = 15;
        description = "How long one cached verdict may be reused.";
      };

      timeoutSecs = mkOption {
        type = types.ints.positive;
        default = 10;
        description = "A probe slower than this is a deny.";
      };

      preempt = mkOption {
        type = types.bool;
        default = false;
        description = ''
          Stop a running adapter when the verdict flips to deny. The job
          returns to `pending` without spending an attempt.
        '';
      };
    };
  };

  profileRouteSubmodule = types.submodule {
    options = {
      endpoint = mkOption {
        type = types.str;
        description = "Logical endpoint name for this route.";
      };

      baseUrl = mkOption {
        type = types.str;
        description = "Non-secret OpenAI-compatible base URL.";
      };

      model = mkOption {
        type = types.str;
        description = "Model identifier sent to the endpoint.";
      };

      healthUrl = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Health URL required when health-aware routing is enabled.";
      };
    };
  };

  profileSubmodule = types.submodule ({ name, ... }: {
    options = {
      workload = mkOption {
        type = types.str;
        default = name;
        description = "Stable generic workload identifier.";
      };

      routing = {
        primary = mkOption {
          type = profileRouteSubmodule;
          description = "Primary endpoint route.";
        };

        fallback = mkOption {
          type = types.nullOr profileRouteSubmodule;
          default = null;
          description = "Optional deterministic fallback route.";
        };

        capability = mkOption {
          type = types.enum [ "chat" "embeddings" "rerank" ];
          description = "Capability required from every selected route.";
        };

        locality = mkOption {
          type = types.enum [ "local-only" "network-allowed" ];
          default = "local-only";
          description = "Network locality permitted by this workload.";
        };

        dataResidency = mkOption {
          type = types.enum [ "local-only" "eu" "ch" "us" "unrestricted" ];
          default = "local-only";
          description = "Data-residency requirement for this workload.";
        };

        healthAware = mkOption {
          type = types.bool;
          default = true;
          description = "Require endpoint health before selecting a route.";
        };

        timeoutSecs = mkOption {
          type = types.ints.positive;
          default = 300;
          description = "Bounded request timeout for the workload route.";
        };

        retry = {
          maxAttempts = mkOption {
            type = types.ints.positive;
            default = 1;
            description = "Maximum route attempts, including the primary.";
          };

          backoffSecs = mkOption {
            type = types.ints.unsigned;
            default = 0;
            description = "Delay between retry/fallback attempts.";
          };
        };

        credentialRequired = mkOption {
          type = types.bool;
          default = false;
          description = "Whether the selected endpoint requires a credential.";
        };
      };

      execution = {
        adapter = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Logical adapter name; command paths stay in adapters.";
        };

        queues = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "Queues the logical adapter may execute.";
        };
      };

      lease = {
        enabled = mkOption {
          type = types.bool;
          default = false;
          description = "Enable durable lease/fencing for this workload.";
        };

        concurrency = mkOption {
          type = types.ints.positive;
          default = 1;
          description = "Maximum concurrent leases for this profile.";
        };

        durationSecs = mkOption {
          type = types.ints.positive;
          default = 900;
          description = "Lease duration in seconds.";
        };

        heartbeatSecs = mkOption {
          type = types.ints.positive;
          default = 30;
          description = "Lease heartbeat interval in seconds.";
        };

        maxAttempts = mkOption {
          type = types.ints.positive;
          default = 3;
          description = "Maximum execution attempts for one desired job.";
        };
      };
    };
  });

  renderRoute = route: {
    endpoint = route.endpoint;
    base_url = route.baseUrl;
    model = route.model;
  } // lib.optionalAttrs (route.healthUrl != null) {
    health_url = route.healthUrl;
  };

  renderProfile = _name: profile: {
    schema_version = 1;
    workload = profile.workload;
    routing = {
      primary = renderRoute profile.routing.primary;
      capability = profile.routing.capability;
      locality = profile.routing.locality;
      data_residency = profile.routing.dataResidency;
      health_aware = profile.routing.healthAware;
      timeout_secs = profile.routing.timeoutSecs;
      retry = {
        max_attempts = profile.routing.retry.maxAttempts;
        backoff_secs = profile.routing.retry.backoffSecs;
      };
      credential_required = profile.routing.credentialRequired;
    } // lib.optionalAttrs (profile.routing.fallback != null) {
      fallback = renderRoute profile.routing.fallback;
    };
    execution = {
      queues = profile.execution.queues;
    } // lib.optionalAttrs (profile.execution.adapter != null) {
      adapter = profile.execution.adapter;
    };
    lease = {
      enabled = profile.lease.enabled;
      concurrency = profile.lease.concurrency;
      duration_secs = profile.lease.durationSecs;
      heartbeat_secs = profile.lease.heartbeatSecs;
      max_attempts = profile.lease.maxAttempts;
    };
  };

  configFile = (pkgs.formats.toml { }).generate "infernix-workerd.toml" ({
    database_url_env = cfg.databaseUrlEnv;
    profiles = lib.mapAttrsToList renderProfile cfg.profiles;
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
    } // lib.optionalAttrs (cfg.gpuDevice != null) {
      gpu_device = cfg.gpuDevice;
    } // lib.optionalAttrs (cfg.admission != null) {
      admission = {
        command = cfg.admission.command;
        recheck_secs = cfg.admission.recheckSecs;
        timeout_secs = cfg.admission.timeoutSecs;
        preempt = cfg.admission.preempt;
      };
    };
    adapters = lib.mapAttrsToList
      (_name: adapter: {
        workload = adapter.workload;
        queues = adapter.queues;
        profiles = adapter.profiles;
        artifacts = adapter.artifacts;
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

    configFile = mkOption {
      type = types.path;
      readOnly = true;
      default = configFile;
      description = "Resolved non-secret worker configuration.";
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

    cacheRoot = mkOption {
      type = types.path;
      default = "/var/lib/infernix-workload/cache";
      description = "Worker-local content-addressed cache root.";
    };

    modelDirectories = mkOption {
      type = types.listOf types.path;
      default = [ ];
      description = ''
        Model and checkpoint directories the worker may read and, for adapters
        that stage weights, write. These join `stagingRoot` and `cacheRoot` in
        the unit's ReadWritePaths.
      '';
    };

    memoryMax = mkOption {
      type = types.str;
      default = "24G";
      description = "MemoryMax for the worker unit: the adapter process tree is killed at this ceiling, not the host.";
    };

    cpuQuota = mkOption {
      type = types.str;
      default = "800%";
      description = "CPUQuota for the worker unit.";
    };

    tasksMax = mkOption {
      type = types.ints.positive;
      default = 8192;
      description = "TasksMax for the worker unit.";
    };

    hubTokenFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = ''
        File holding the `procedure` hub token. Loaded as the systemd
        credential `infernix-hub-token`; workerd reads it through
        `INFERNIX_HUB_TOKEN_FILE` and never puts it in the adapter envelope.
      '';
    };

    gpuDevice = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "atlas-7900xtx";
      description = ''
        Device id whose gate lock (`/run/infernix/gpu/<device>.lock`) this
        worker's adapters hold. Created with group `render`, mode 0660.
      '';
    };

    admission = mkOption {
      type = types.nullOr admissionSubmodule;
      default = null;
      description = ''
        Optional host admission probe. While it denies, the worker claims no
        jobs; with `preempt` a running adapter is stopped and its job returns
        to `pending`. The probe command is host-specific (for example a Canix
        CLI check); the worker only reads its exit status.
      '';
    };

    adapters = mkOption {
      type = types.attrsOf adapterSubmodule;
      default = { };
      description = "Workload adapters available to this worker.";
    };

    profiles = mkOption {
      type = types.attrsOf profileSubmodule;
      default = { };
      description = ''
        Generic typed routing, execution, and lease profiles. Profile output
        contains only endpoint facts and logical adapter names; credentials,
        source, prompts, and executable command lines are not serialized here.
      '';
    };
  };

  options.services.infernix.gpuGate = {
    enable = mkEnableOption "the shared Infernix GPU device gate for non-ledger GPU tenants";

    device = mkOption {
      type = types.str;
      default = config.networking.hostName;
      description = "Device id whose lock file tenants take shared (for example `atlas-7900xtx`).";
    };

    lockPath = mkOption {
      type = types.str;
      readOnly = true;
      default = gpuGate.lockPath gpuGateCfg.device;
      description = "Resolved per-device gate lock. Adapters take it exclusively.";
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
      {
        assertion = cfg.admission == null || cfg.admission.command != [ ];
        message = "services.infernix.workloadFabric.admission.command must not be empty.";
      }
    ]
    ++ lib.mapAttrsToList
      (name: adapter: {
        assertion = adapter.queues != [ ];
        message = "services.infernix.workloadFabric.adapters.${name}.queues must not be empty.";
      })
      cfg.adapters
    ++ lib.mapAttrsToList
      (name: adapter: {
        assertion = adapter.profiles != [ ];
        message = "services.infernix.workloadFabric.adapters.${name}.profiles must not be empty: a job's profile:<id> requirement selects the adapter.";
      })
      cfg.adapters
    ++ lib.concatLists (lib.mapAttrsToList
      (name: profile: [
        {
          assertion = !profile.routing.healthAware || profile.routing.primary.healthUrl != null;
          message = "services.infernix.workloadFabric.profiles.${name} requires a primary healthUrl for health-aware routing.";
        }
        {
          assertion = profile.routing.fallback == null || profile.routing.fallback.healthUrl != null || !profile.routing.healthAware;
          message = "services.infernix.workloadFabric.profiles.${name} requires a fallback healthUrl for health-aware routing.";
        }
        {
          assertion = profile.routing.fallback == null || profile.routing.retry.maxAttempts > 1;
          message = "services.infernix.workloadFabric.profiles.${name}.routing.retry.maxAttempts must be greater than one when fallback is configured.";
        }
        {
          assertion = profile.execution.adapter != null || profile.execution.queues == [ ];
          message = "services.infernix.workloadFabric.profiles.${name}.execution.queues requires an adapter.";
        }
        {
          assertion = !profile.lease.enabled || profile.execution.adapter != null;
          message = "services.infernix.workloadFabric.profiles.${name}.lease.enabled requires an execution adapter.";
        }
        {
          assertion = !profile.lease.enabled || profile.execution.queues != [ ];
          message = "services.infernix.workloadFabric.profiles.${name}.lease.enabled requires an execution queue.";
        }
        {
          assertion = !profile.lease.enabled || profile.lease.heartbeatSecs < profile.lease.durationSecs;
          message = "services.infernix.workloadFabric.profiles.${name}.lease.heartbeatSecs must be less than durationSecs.";
        }
      ])
      cfg.profiles);

    users.groups.${workerUser} = { };

    users.users.${workerUser} = {
      isSystemUser = true;
      group = workerUser;
      # The render group owns the DRM nodes; video owns the encoder devices.
      extraGroups = [ "render" "video" ];
      home = "/var/lib/infernix-workload";
      description = "Infernix workload adapter user";
    };

    systemd.tmpfiles.rules = lib.unique (
      [
        "d ${toString cfg.stagingRoot} 0750 ${workerUser} ${workerUser} -"
        "d ${toString cfg.cacheRoot} 0750 ${workerUser} ${workerUser} -"
      ]
      ++ lib.optional gpuGateCfg.enable (gpuGate.tmpfilesRule gpuGateCfg.device)
      ++ lib.optional (cfg.gpuDevice != null) (gpuGate.tmpfilesRule cfg.gpuDevice)
    );

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
        User = workerUser;
        Group = workerUser;
        EnvironmentFile = databaseEnvironmentFile;
        Environment = databaseEnvironment
          ++ lib.optional (cfg.hubTokenFile != null)
            "INFERNIX_HUB_TOKEN_FILE=/run/credentials/infernix-workerd.service/infernix-hub-token"
          ++ lib.optional (cfg.gpuDevice != null) "INFERNIX_GPU_DEVICE=${cfg.gpuDevice}"
          ++ [ "INFERNIX_ADAPTER_CACHE_DIR=${toString cfg.cacheRoot}" ];
        LoadCredential =
          lib.optional (cfg.hubTokenFile != null) "infernix-hub-token:${cfg.hubTokenFile}";
        StateDirectory = "infernix-workload";
        # Containment: adapter processes are killed at the unit's ceiling, not
        # the host's.
        MemoryMax = cfg.memoryMax;
        CPUQuota = cfg.cpuQuota;
        TasksMax = cfg.tasksMax;
        ReadWritePaths = [cfg.stagingRoot cfg.cacheRoot] ++ map toString cfg.modelDirectories;
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      };
    };
  };
}
