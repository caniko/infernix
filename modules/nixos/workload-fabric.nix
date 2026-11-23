{ config
, infernixSelf
, lib
, pkgs
, ...
}:
let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.workloadFabric;
  imageRunnerCfg = config.services.infernix.imageRunner;
  gpuGateCfg = config.services.infernix.gpuGate;
  gpuGate = import ../../lib/gpu-gate.nix {inherit lib pkgs;};
  workerUser = "infernix-worker";
  imageRunnerUser = "infernix-image-runner";
  handoffGroup = "infernix-image-handoff";

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

      deliveryMode = mkOption {
        type = types.enum [ "generations" "pack_output" ];
        default = "generations";
        description = ''
          Artifact delivery route: generic `/v1/generations/{id}/artifacts`
          (`generations`) or pack-scoped `PUT /v1/p/{pack}/worker/jobs/{id}/output`
          (`pack_output`).
        '';
      };

      deliveryPack = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Pack name when deliveryMode is `pack_output` (e.g. `vagrant`).";
      };

      tokenFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = ''
          Separate worker credential file loaded as a dedicated systemd
          credential for this adapter (required when deliveryMode is
          `pack_output` so pack tokens never mix with the generic procedure token).
        '';
      };

      preemptRequeue = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Whether admission preemption returns a running job to `pending`
          without spending an attempt. Set `false` for non-idempotent
          post-submit adapters (`max_attempts = 1`).
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

  adapterTokenEnv = name:
    "INFERNIX_ADAPTER_TOKEN_FILE_${lib.toUpper (lib.replaceStrings [ "-" "." ] [ "_" "_" ] name)}";

  adapterTokenCredential = name: "infernix-adapter-token-${name}";

  adaptersWithToken = lib.filterAttrs (_: adapter: adapter.tokenFile != null) cfg.adapters;

  validAdapterName = name: builtins.match "^[a-z0-9][a-z0-9_-]*$" name != null;

  generatedTokenEnvs = map adapterTokenEnv (builtins.attrNames adaptersWithToken);
  generatedTokenCredentials = map adapterTokenCredential (builtins.attrNames adaptersWithToken);

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
    } // lib.optionalAttrs (cfg.hubUrl != null) {
      hub_url = cfg.hubUrl;
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
      (name: adapter: {
        workload = adapter.workload;
        queues = adapter.queues;
        profiles = adapter.profiles;
        artifacts = adapter.artifacts;
        command = adapter.command;
        args = adapter.args;
        environment = adapter.environment;
      } // lib.optionalAttrs (adapter.workingDirectory != null) {
        working_directory = toString adapter.workingDirectory;
      } // lib.optionalAttrs (adapter.deliveryMode != "generations") {
        delivery_mode = adapter.deliveryMode;
      } // lib.optionalAttrs (adapter.deliveryPack != null) {
        delivery_pack = adapter.deliveryPack;
      } // lib.optionalAttrs (adapter.tokenFile != null) {
        hub_token_file_env = adapterTokenEnv name;
      } // lib.optionalAttrs (!adapter.preemptRequeue) {
        preempt_requeue = false;
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

    memoryHigh = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Optional MemoryHigh for the worker unit.";
    };

    memoryMax = mkOption {
      type = types.str;
      default = "24G";
      description = "MemoryMax for the worker unit: the adapter process tree is killed at this ceiling, not the host.";
    };

    memorySwapMax = mkOption {
      type = types.str;
      default = "0";
      description = "MemorySwapMax for the worker unit (defaults to 0 so adapters never thrash swap).";
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

    hubUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = ''
        Trusted Hub base origin (`http(s)://host[:port]`) used for pack-scoped
        input downloads and output uploads. Required when any adapter uses
        `deliveryMode = "pack_output"` so job payloads can never redirect
        worker credentials or private pixels to attacker-controlled origins.
      '';
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

  options.services.infernix.imageRunner = {
    enable = mkEnableOption "the isolated, credential-free Infernix host-local image runner service";

    package = mkOption {
      type = types.package;
      default = infernixSelf.packages.${pkgs.stdenv.hostPlatform.system}.infernix-image-run;
      defaultText = lib.literalExpression "inputs.infernix.packages.<system>.infernix-image-run";
      description = "Package providing the credential-free `infernix-image-run` binary.";
    };

    socketPath = mkOption {
      type = types.str;
      default = "/run/infernix-image-runner/runner.sock";
      description = "Unix socket where the isolated image runner accepts local requests.";
    };

    lockPath = mkOption {
      type = types.str;
      default = "/run/infernix-image-runner/image-runner.lock";
      description = "Host-local serial runner mutex lock path.";
    };

    registryDir = mkOption {
      type = types.path;
      description = "Immutable directory containing `registry.json` and registered graph templates.";
    };

    stagingRoot = mkOption {
      type = types.path;
      default = cfg.stagingRoot;
      defaultText = lib.literalExpression "config.services.infernix.workloadFabric.stagingRoot";
      description = "Shared staging root accessible via the dedicated handoff group.";
    };

    modelDirectories = mkOption {
      type = types.listOf types.path;
      default = [ ];
      description = "Read-only provisioned model directories accessible to the image runner.";
    };

    comfyUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Loopback ComfyUI base URL.";
    };

    comfyUnit = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Systemd unit name for ComfyUI cgroup/invocation/--disable-metadata checks.";
    };

    comfyCleanupSocket = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Authorized ComfyUI-identity cleanup Unix socket path.";
    };

    comfyCleanupCommand = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Optional ownership-scoped cleanup command for ComfyUI input/output files.";
    };

    comfyRequireCleanup = mkOption {
      type = types.bool;
      default = true;
      description = "Require an authorized ComfyUI file cleanup route when ComfyUI is configured.";
    };

    llamaSwapUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Optional llama-swap loopback URL for best-effort unload before GPU use.";
    };

    gpuLockPath = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Exclusive per-device GPU lock path used by Qwen streaming.";
    };

    backend = mkOption {
      type = types.nullOr (types.enum [ "rocm" "cuda" ]);
      default = null;
      description = "GPU backend (`rocm` or `cuda`) for VRAM checks and Qwen execution.";
    };

    vramFloorMib = mkOption {
      type = types.nullOr types.ints.positive;
      default = null;
      description = "Free VRAM admission floor in MiB.";
    };

    memAdmitGib = mkOption {
      type = types.ints.positive;
      default = 40;
      description = "Minimum host MemAvailable (GiB) to admit a job.";
    };

    memTripGib = mkOption {
      type = types.ints.positive;
      default = 16;
      description = "Minimum host MemAvailable (GiB) while running before watchdog trips.";
    };

    psiAdmitAvg10 = mkOption {
      type = types.float;
      default = 5.0;
      description = "Maximum full memory PSI avg10 (%) to admit a job.";
    };

    psiTripAvg10 = mkOption {
      type = types.float;
      default = 10.0;
      description = "Maximum full memory PSI avg10 (%) while running before watchdog trips.";
    };

    admissionCommand = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Optional host admission probe command.";
    };

    leaseHook = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Optional local acquire/recheck/release lease hook command.";
    };

    leaseRequired = mkOption {
      type = types.bool;
      default = false;
      description = "Fail closed unless a local lease hook is configured and admits the job.";
    };

    qwen = mkOption {
      type = types.nullOr (types.submodule {
        options = {
          python = mkOption {
            type = types.path;
            description = "Packaged Qwen adapter Python executable.";
          };
          adapterScript = mkOption {
            type = types.path;
            description = "Packaged Qwen adapter.py path.";
          };
          denoiserDir = mkOption {
            type = types.path;
            description = "Provisioned Qwen denoiser directory.";
          };
          snapshotDir = mkOption {
            type = types.path;
            description = "Provisioned Qwen snapshot directory.";
          };
          lightningDir = mkOption {
            type = types.path;
            description = "Provisioned Qwen Lightning LoRA directory.";
          };
        };
      });
      default = null;
      description = "Optional Qwen Image Edit 2511 streaming adapter configuration.";
    };

    memoryHigh = mkOption {
      type = types.str;
      default = "25G";
      description = "MemoryHigh for the runner unit (Qwen children get per-profile 24G sibling cgroups).";
    };

    memoryMax = mkOption {
      type = types.str;
      default = "26G";
      description = "MemoryMax for the runner unit (Qwen children get per-profile 24G sibling cgroups).";
    };

    memorySwapMax = mkOption {
      type = types.str;
      default = "0";
      description = "MemorySwapMax for the runner unit.";
    };
  };

  config = lib.mkMerge [
  (mkIf cfg.enable {
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
      {
        assertion =
          cfg.admission == null
          || cfg.heartbeatIntervalSecs + cfg.admission.timeoutSecs < cfg.leaseDurationSecs;
        message = "services.infernix.workloadFabric.heartbeatIntervalSecs + admission.timeoutSecs must be less than leaseDurationSecs: the probe runs beside the heartbeat, so one probe must fit inside the lease.";
      }
      {
        assertion = lib.length (lib.unique generatedTokenEnvs) == lib.length generatedTokenEnvs;
        message = "services.infernix.workloadFabric.adapters: generated adapter token environment variable names collide (e.g. hyphen vs underscore).";
      }
      {
        assertion = lib.length (lib.unique generatedTokenCredentials) == lib.length generatedTokenCredentials;
        message = "services.infernix.workloadFabric.adapters: generated systemd credential names collide.";
      }
    ]
    ++ lib.mapAttrsToList
      (name: _: {
        assertion = validAdapterName name;
        message = "services.infernix.workloadFabric.adapters.${name}: adapter name must match ^[a-z0-9][a-z0-9_-]*$.";
      })
      cfg.adapters
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
    ++ lib.mapAttrsToList
      (name: adapter: {
        assertion = adapter.deliveryMode != "pack_output" || (adapter.artifacts && adapter.deliveryPack != null && adapter.tokenFile != null && cfg.hubUrl != null);
        message = "services.infernix.workloadFabric.adapters.${name}: deliveryMode = pack_output requires artifacts = true, deliveryPack, a separate tokenFile, and services.infernix.workloadFabric.hubUrl.";
      })
      cfg.adapters
    ++ lib.mapAttrsToList
      (name: adapter: {
        assertion = adapter.deliveryMode != "pack_output" || (cfg.hubTokenFile == null || toString adapter.tokenFile != toString cfg.hubTokenFile);
        message = "services.infernix.workloadFabric.adapters.${name}: pack_output tokenFile must be distinct from services.infernix.workloadFabric.hubTokenFile.";
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
    users.groups.${handoffGroup} = lib.mkIf imageRunnerCfg.enable { };

    users.users.${workerUser} = {
      isSystemUser = true;
      group = workerUser;
      # The render group owns the DRM nodes; video owns the encoder devices.
      extraGroups = [ "render" "video" ] ++ lib.optional imageRunnerCfg.enable handoffGroup;
      home = "/var/lib/infernix-workload";
      description = "Infernix workload adapter user";
    };

    systemd.tmpfiles.rules = lib.unique (
      [
        (if imageRunnerCfg.enable
          then "d ${toString cfg.stagingRoot} 2770 ${workerUser} ${handoffGroup} -"
          else "d ${toString cfg.stagingRoot} 0750 ${workerUser} ${workerUser} -")
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
      after = [ "network-online.target" "infernix-workload-migrate.service" ]
        ++ lib.optional imageRunnerCfg.enable "infernix-image-runner.service";
      wants = [ "network-online.target" ]
        ++ lib.optional imageRunnerCfg.enable "infernix-image-runner.service";
      requires = [ "infernix-workload-migrate.service" ];
      serviceConfig = {
        ExecStart = "${workerExec} --config ${configFile} worker";
        User = workerUser;
        Group = workerUser;
        EnvironmentFile = databaseEnvironmentFile;
        Environment = databaseEnvironment
          ++ lib.optional (cfg.hubTokenFile != null)
            "INFERNIX_HUB_TOKEN_FILE=/run/credentials/infernix-workerd.service/infernix-hub-token"
          ++ lib.mapAttrsToList
            (name: _: "${adapterTokenEnv name}=/run/credentials/infernix-workerd.service/${adapterTokenCredential name}")
            adaptersWithToken
          ++ lib.optional imageRunnerCfg.enable "INFERNIX_IMAGE_RUNNER_SOCKET=${imageRunnerCfg.socketPath}"
          ++ lib.optional (cfg.gpuDevice != null) "INFERNIX_GPU_DEVICE=${cfg.gpuDevice}"
          ++ [ "INFERNIX_ADAPTER_CACHE_DIR=${toString cfg.cacheRoot}" ];
        LoadCredential =
          lib.optional (cfg.hubTokenFile != null) "infernix-hub-token:${cfg.hubTokenFile}"
          ++ lib.mapAttrsToList
            (name: adapter: "${adapterTokenCredential name}:${adapter.tokenFile}")
            adaptersWithToken;
        StateDirectory = "infernix-workload";
        # Containment: adapter processes are killed at the unit's ceiling, not
        # the host's.
        MemoryMax = cfg.memoryMax;
        MemorySwapMax = cfg.memorySwapMax;
        CPUQuota = cfg.cpuQuota;
        TasksMax = cfg.tasksMax;
        ReadWritePaths = [cfg.stagingRoot cfg.cacheRoot] ++ map toString cfg.modelDirectories;
        Restart = "on-failure";
        RestartSec = 5;
        KillMode = "mixed";
        # Coupled to workerd's default cancel_grace_secs (30), plus 30s for lease release.
        TimeoutStopSec = 60;
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      } // lib.optionalAttrs (cfg.memoryHigh != null) {
        MemoryHigh = cfg.memoryHigh;
      };
    };
  })

  (mkIf imageRunnerCfg.enable (
    let
      runnerArgs = [
        "--serve-socket" imageRunnerCfg.socketPath
        "--registry" (toString imageRunnerCfg.registryDir)
        "--runner-lock" imageRunnerCfg.lockPath
        "--mem-admit-gib" (toString imageRunnerCfg.memAdmitGib)
        "--mem-trip-gib" (toString imageRunnerCfg.memTripGib)
        "--psi-admit-avg10" (toString imageRunnerCfg.psiAdmitAvg10)
        "--psi-trip-avg10" (toString imageRunnerCfg.psiTripAvg10)
      ]
      ++ lib.optionals (imageRunnerCfg.comfyUrl != null) [ "--comfy-url" imageRunnerCfg.comfyUrl ]
      ++ lib.optionals (imageRunnerCfg.comfyUnit != null) [ "--comfy-unit" imageRunnerCfg.comfyUnit ]
      ++ lib.optionals (imageRunnerCfg.comfyCleanupSocket != null) [ "--comfy-cleanup-socket" imageRunnerCfg.comfyCleanupSocket ]
      ++ lib.concatMap (a: [ "--comfy-cleanup-cmd" a ]) imageRunnerCfg.comfyCleanupCommand
      ++ lib.optional (imageRunnerCfg.comfyUrl != null && imageRunnerCfg.comfyRequireCleanup) "--comfy-require-cleanup"
      ++ lib.optionals (imageRunnerCfg.llamaSwapUrl != null) [ "--llama-swap-url" imageRunnerCfg.llamaSwapUrl ]
      ++ lib.optionals (imageRunnerCfg.gpuLockPath != null) [ "--gpu-lock" (toString imageRunnerCfg.gpuLockPath) ]
      ++ lib.optionals (imageRunnerCfg.backend != null) [ "--backend" imageRunnerCfg.backend ]
      ++ lib.optionals (imageRunnerCfg.vramFloorMib != null) [ "--vram-floor-mib" (toString imageRunnerCfg.vramFloorMib) ]
      ++ lib.concatMap (a: [ "--admission-cmd" a ]) imageRunnerCfg.admissionCommand
      ++ lib.concatMap (a: [ "--lease-hook" a ]) imageRunnerCfg.leaseHook
      ++ lib.optional imageRunnerCfg.leaseRequired "--lease-required"
      ++ lib.optionals (imageRunnerCfg.qwen != null) ([
        "--qwen-python" (toString imageRunnerCfg.qwen.python)
        "--qwen-adapter" (toString imageRunnerCfg.qwen.adapterScript)
        "--qwen-denoiser" (toString imageRunnerCfg.qwen.denoiserDir)
        "--qwen-snapshot" (toString imageRunnerCfg.qwen.snapshotDir)
        "--qwen-lightning" (toString imageRunnerCfg.qwen.lightningDir)
      ] ++ lib.optionals (imageRunnerCfg.backend != null) [ "--qwen-backend" imageRunnerCfg.backend ]);
      socketDir = builtins.dirOf imageRunnerCfg.socketPath;
    in {
      assertions = [
        {
          assertion = imageRunnerCfg.comfyUrl == null || imageRunnerCfg.comfyUnit != null;
          message = "services.infernix.imageRunner: comfyUrl requires comfyUnit so cgroup limits, InvocationID, and --disable-metadata can be verified.";
        }
        {
          assertion =
            imageRunnerCfg.comfyUrl == null
            || !imageRunnerCfg.comfyRequireCleanup
            || imageRunnerCfg.comfyCleanupSocket != null
            || imageRunnerCfg.comfyCleanupCommand != [ ];
          message = "services.infernix.imageRunner: comfyUrl with comfyRequireCleanup requires comfyCleanupSocket or comfyCleanupCommand.";
        }
      ];

      users.groups.${imageRunnerUser} = { };
      users.groups.${handoffGroup} = { };

      users.users.${imageRunnerUser} = {
        isSystemUser = true;
        group = imageRunnerUser;
        extraGroups = [ "render" "video" handoffGroup ];
        home = "/var/empty";
        description = "Infernix isolated credential-free image execution runner";
      };

      systemd.tmpfiles.rules = [
        "d ${socketDir} 2750 ${imageRunnerUser} ${handoffGroup} -"
      ];

      # Isolated, credential-free execution core: separate UID, separate mount
      # namespace (no access to /run/credentials/infernix-workerd.service), and
      # delegated cgroup v2 subtree so each Qwen child runs in its own sibling
      # cgroup with MemoryHigh/MemoryMax/MemorySwapMax=0/OOMPolicy=kill.
      systemd.services.infernix-image-runner = {
        description = "Infernix isolated host-local image runner";
        wantedBy = [ "multi-user.target" ];
        after = [ "local-fs.target" "systemd-tmpfiles-setup.service" ]
          ++ lib.optional (imageRunnerCfg.comfyCleanupSocket != null) "infernix-comfyui-cleanup.service";
        serviceConfig = {
          ExecStart = "${imageRunnerCfg.package}/bin/infernix-image-run ${lib.escapeShellArgs runnerArgs}";
          User = imageRunnerUser;
          Group = imageRunnerUser;
          SupplementaryGroups = [ "render" "video" handoffGroup ];
          UMask = "0007";
          Delegate = "memory pids";
          DelegateSubgroup = "runner";
          MemoryHigh = imageRunnerCfg.memoryHigh;
          MemoryMax = imageRunnerCfg.memoryMax;
          MemorySwapMax = imageRunnerCfg.memorySwapMax;
          ReadWritePaths = [ imageRunnerCfg.stagingRoot socketDir ]
            ++ lib.optional (imageRunnerCfg.gpuLockPath != null) (builtins.dirOf (toString imageRunnerCfg.gpuLockPath))
            ++ lib.optional (imageRunnerCfg.comfyCleanupSocket != null) (builtins.dirOf imageRunnerCfg.comfyCleanupSocket);
          ReadOnlyPaths = [ imageRunnerCfg.registryDir ] ++ map toString imageRunnerCfg.modelDirectories;
          Restart = "on-failure";
          RestartSec = 5;
          KillMode = "control-group";
          TimeoutStopSec = 60;
          NoNewPrivileges = true;
          ProtectSystem = "strict";
          ProtectHome = true;
          PrivateTmp = true;
        };
      };
    }
  ))
  ];
}
