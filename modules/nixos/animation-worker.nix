# Animation workload worker: claims jobs for this host's GPU from the
# PostgreSQL ledger, provisions pinned uv adapter environments and verified
# model checkpoints once via `infernix-animation-provision`, and gates GPU
# claims through `services.infernix.gpuAdmission`.
{
  config,
  infernixSelf,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.infernix.animationWorker;
  assembleDatabaseEnv = cfg.databaseUrlFile == null;
  generatedDatabaseEnvFile = "/run/infernix-workload/database.env";
  effectiveDatabaseUrlFile =
    if assembleDatabaseEnv
    then generatedDatabaseEnvFile
    else toString cfg.databaseUrlFile;
  python = cfg.pythonPackage;
  pythonBin = "${python}/bin/python${python.pythonVersion}";

  provision = pkgs.writeShellApplication {
    name = "infernix-animation-provision";
    runtimeInputs = [
      pkgs.coreutils
      pkgs.findutils
      pkgs.git
      cfg.huggingfacePackage
      pkgs.jq
      pkgs.uv
      python
    ];
    text = ''
      set -eu
      model_root=${lib.escapeShellArg cfg.modelRoot}
      profiles_dir=${lib.escapeShellArg (
        if cfg.profilesDir == null
        then ""
        else toString cfg.profilesDir
      )}
      want=${lib.escapeShellArg cfg.provisionStamp}

      export UV_CACHE_DIR="$model_root/.uv-cache"
      export UV_LINK_MODE=copy
      export UV_PYTHON_DOWNLOADS=never
      export HF_HOME="$model_root/.hf"
      export CC=${lib.escapeShellArg "${pkgs.stdenv.cc}/bin/cc"}
      export CXX=${lib.escapeShellArg "${pkgs.stdenv.cc}/bin/c++"}
      # huggingface_hub 2.x installs `hf`; keep the old name as a fallback.
      hf_cli=$(command -v hf || command -v huggingface-cli)

      mkdir -p "$model_root/.venvs" "$model_root/.provisioned" "$model_root/weights" "$model_root/code"

      sync_adapter() {
        name=$1 project=$2
        marker="$model_root/.provisioned/$name"
        if [ -f "$marker" ] && [ "$(cat "$marker")" = "$want" ]; then
          echo "$name: environment current"
          return 0
        fi
        echo "$name: uv sync --frozen ($project)"
        UV_PROJECT_ENVIRONMENT="$model_root/.venvs/$name" \
          uv sync --frozen --python ${lib.escapeShellArg pythonBin} \
          --project "$project"
        printf '%s\n' "$want" > "$marker"
      }

      ${builtins.readFile ../../lib/animation-prefetch.sh}

      ${lib.concatStrings (lib.mapAttrsToList (name: adapter: ''
          sync_adapter ${lib.escapeShellArgs [name "${cfg.adaptersRoot}/${adapter.project}"]}
        '')
        cfg.adapters)}
      ${cfg.extraProvision}
      ${lib.optionalString (cfg.profilesDir != null) (
        lib.concatMapStrings (id: "prefetch_profile ${lib.escapeShellArg id}\n")
        (lib.concatMap (adapter: adapter.profiles) (lib.attrValues cfg.adapters))
      )}
    '';
  };

  qwenProvision = pkgs.writeShellApplication {
    name = "infernix-qwen-image-provision";
    runtimeInputs = [
      pkgs.coreutils
      pkgs.uv
      python
    ];
    text = ''
      set -eu
      model_root=${lib.escapeShellArg cfg.modelRoot}
      project=${lib.escapeShellArg "${cfg.adaptersRoot}/${cfg.qwenProvisionProject}"}
      want=${lib.escapeShellArg cfg.provisionStamp}
      venv="$model_root/.venvs/qwen-image-edit-streaming"
      marker="$model_root/.provisioned/qwen-image-edit-streaming"

      export UV_CACHE_DIR="$model_root/.uv-cache"
      export UV_LINK_MODE=copy
      export UV_PYTHON_DOWNLOADS=never
      export CC=${lib.escapeShellArg "${pkgs.stdenv.cc}/bin/cc"}
      export CXX=${lib.escapeShellArg "${pkgs.stdenv.cc}/bin/c++"}

      mkdir -p "$model_root/.venvs" "$model_root/.provisioned"
      if [ -f "$marker" ] && [ "$(cat "$marker")" = "$want" ]; then
        echo "qwen-image-edit-streaming: environment current"
      else
        echo "qwen-image-edit-streaming: uv sync --frozen ($project)"
        UV_PROJECT_ENVIRONMENT="$venv" \
          uv sync --frozen --python ${lib.escapeShellArg pythonBin} \
          --project "$project"
        printf '%s\n' "$want" > "$marker"
      fi
      chgrp -R infernix-image-handoff "$venv"
      chmod -R g+rX,g-w,o-rwx "$venv"
    '';
  };

  # The DSN is assembled at unit start so the password never reaches the store.
  databaseEnv = pkgs.writeShellApplication {
    name = "infernix-workload-database-env";
    runtimeInputs = [pkgs.coreutils];
    text = ''
      set -eu
      install -d -m 0750 -o infernix-worker -g infernix-worker /run/infernix-workload
      umask 077
      {
        printf 'INFERNIX_WORKLOAD_DATABASE_URL=host=%s port=%s user=%s dbname=%s sslmode=%s password=' \
          ${lib.escapeShellArg cfg.databaseHost} \
          ${lib.escapeShellArg (toString cfg.databasePort)} \
          ${lib.escapeShellArg cfg.databaseUser} \
          ${lib.escapeShellArg cfg.databaseName} \
          ${lib.escapeShellArg cfg.databaseSslMode}
        cat ${lib.escapeShellArg (toString cfg.databasePasswordFile)}
        printf '\n'
      } > ${lib.escapeShellArg generatedDatabaseEnvFile}
      chown infernix-worker:infernix-worker ${lib.escapeShellArg generatedDatabaseEnvFile}
      chmod 0600 ${lib.escapeShellArg generatedDatabaseEnvFile}
    '';
  };

  # Only delivered or failed work loses its staging: workerd keeps
  # succeeded-but-undelivered jobs so it can finish their uploads.
  janitorSql = pkgs.writeText "infernix-staging-janitor.sql" ''
    SELECT id FROM infernix_workload_jobs
     WHERE lease_owner = :'worker'
       AND updated_at < now() - interval '${toString cfg.stagingJanitor.retentionDays} days'
       AND (state IN ('failed', 'cancelled')
            OR (state = 'succeeded' AND delivered_at IS NOT NULL));
  '';

  janitor = pkgs.writeShellApplication {
    name = "infernix-staging-janitor";
    runtimeInputs = [pkgs.coreutils pkgs.gnugrep cfg.postgresqlPackage];
    text = ''
      set -eu
      staging=${lib.escapeShellArg "${cfg.modelRoot}/staging"}
      psql "$INFERNIX_WORKLOAD_DATABASE_URL" -tA \
        -v worker=${lib.escapeShellArg cfg.workerName} -f ${janitorSql} |
        while IFS= read -r id; do
          printf '%s' "$id" | grep -Eq '^[A-Za-z0-9_-]+$' || {
            echo "janitor: refusing job id '$id'" >&2
            continue
          }
          rm -rf -- "''${staging:?}/$id"
          echo "janitor: removed staging for $id"
        done
    '';
  };
in {
  options.services.infernix.animationWorker = {
    enable = lib.mkEnableOption "the animation workload worker";

    device = lib.mkOption {
      type = lib.types.str;
      description = "GPU device id; jobs require `gpu:<device>`.";
      example = "worker-gpu0";
    };

    modelRoot = lib.mkOption {
      type = lib.types.str;
      description = "Root for adapter environments, weights, cache and staging.";
    };

    profilesDir = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional directory containing `<model>/<id>.json` model profile pins for checkpoint prefetch.";
    };

    adaptersRoot = lib.mkOption {
      type = lib.types.path;
      default = "${infernixSelf}/adapters";
      defaultText = lib.literalExpression ''"''${inputs.infernix}/adapters"'';
      description = "Directory containing pinned uv adapter projects.";
    };

    provisionStamp = lib.mkOption {
      type = lib.types.str;
      default = toString infernixSelf;
      defaultText = lib.literalExpression "toString inputs.infernix";
      description = "Marker stamp written after `uv sync --frozen` so environments re-sync when inputs change.";
    };

    pythonPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.python312;
      defaultText = lib.literalExpression "pkgs.python312";
      description = "Python interpreter passed to `uv sync --frozen`.";
    };

    huggingfacePackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.huggingface-cli or pkgs.python3Packages.huggingface-hub;
      defaultText = lib.literalExpression "pkgs.huggingface-cli or pkgs.python3Packages.huggingface-hub";
      description = "Package providing `hf` or `huggingface-cli` for checkpoint prefetch.";
    };

    postgresqlPackage = lib.mkOption {
      type = lib.types.package;
      default = pkgs.postgresql_18;
      defaultText = lib.literalExpression "pkgs.postgresql_18";
      description = "PostgreSQL client package used by the staging janitor.";
    };

    workerName = lib.mkOption {
      type = lib.types.str;
      default = config.networking.hostName;
      defaultText = lib.literalExpression "config.networking.hostName";
      description = "Worker lease owner name queried by the staging janitor.";
    };

    memoryMax = lib.mkOption {
      type = lib.types.str;
      default = "24G";
      description = "MemoryMax limit for the workload worker service.";
    };

    cpuQuota = lib.mkOption {
      type = lib.types.str;
      default = "800%";
      description = "CPUQuota limit for the workload worker service.";
    };

    tasksMax = lib.mkOption {
      type = lib.types.int;
      default = 4096;
      description = "TasksMax limit for the workload worker service.";
    };

    concurrency = lib.mkOption {
      type = lib.types.int;
      default = 1;
      description = "Number of concurrent jobs claimed by this worker.";
    };

    runMigrations = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = ''
        Whether this host runs `infernix-workload-migrate.service`. Remote
        workers leave this false because the ledger host owns the schema.
      '';
    };

    databaseUrlFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = ''
        Optional pre-assembled EnvironmentFile containing
        `INFERNIX_WORKLOAD_DATABASE_URL=...`. When unset, the module assembles
        `/run/infernix-workload/database.env` from `databaseHost` and
        `databasePasswordFile`.
      '';
    };

    databaseHost = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "PostgreSQL ledger host used when `databaseUrlFile` is unset.";
    };

    databasePort = lib.mkOption {
      type = lib.types.port;
      default = 5432;
      description = "PostgreSQL ledger port used when `databaseUrlFile` is unset.";
    };

    databaseName = lib.mkOption {
      type = lib.types.str;
      default = "infernix_workload";
      description = "PostgreSQL database name used when `databaseUrlFile` is unset.";
    };

    databaseUser = lib.mkOption {
      type = lib.types.str;
      default = "infernix_workload_${config.networking.hostName}";
      defaultText = lib.literalExpression ''"infernix_workload_''${config.networking.hostName}"'';
      description = "PostgreSQL user used when `databaseUrlFile` is unset.";
    };

    databaseSslMode = lib.mkOption {
      type = lib.types.str;
      default = "disable";
      description = "PostgreSQL sslmode used when `databaseUrlFile` is unset.";
    };

    databasePasswordFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Path to a root-readable file containing the PostgreSQL password when `databaseUrlFile` is unset.";
    };

    hubTokenFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional path to the hub procedure bearer token file passed to `workloadFabric.hubTokenFile`.";
    };

    enableGpuAdmission = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable `services.infernix.gpuAdmission` and wire its `checkCommand` into `workloadFabric.admission`.";
    };

    adapters = lib.mkOption {
      default = {};
      description = "Adapters this worker serves, keyed by adapter name.";
      type = lib.types.attrsOf (lib.types.submodule {
        options = {
          project = lib.mkOption {
            type = lib.types.str;
            description = "uv project under `adaptersRoot` (e.g. `sam2` or `sam2/xpu`).";
          };
          script = lib.mkOption {
            type = lib.types.str;
            description = "Adapter entry point under `adaptersRoot`, e.g. `sam2/adapter.py`.";
          };
          queues = lib.mkOption {type = lib.types.listOf lib.types.str;};
          profiles = lib.mkOption {type = lib.types.listOf lib.types.str;};
          environment = lib.mkOption {
            type = lib.types.attrsOf lib.types.str;
            default = {};
          };
        };
      });
    };

    extraProvision = lib.mkOption {
      type = lib.types.lines;
      default = "";
      description = "Extra provisioning shell run after the environments sync (e.g. a pinned code checkout).";
    };

    qwenProvisionProject = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Optional uv project path under `adaptersRoot` provisioned in a dedicated `infernix-qwen-image-provision` unit.";
    };

    stagingJanitor = {
      enable = lib.mkOption {
        type = lib.types.bool;
        default = true;
        description = "Run a daily timer to prune delivered or failed staging directories.";
      };

      retentionDays = lib.mkOption {
        type = lib.types.ints.positive;
        default = 7;
        description = "Days to retain delivered, failed or cancelled staging directories.";
      };

      onCalendar = lib.mkOption {
        type = lib.types.str;
        default = "daily";
        description = "systemd OnCalendar schedule for the staging janitor timer.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.databaseUrlFile != null || (cfg.databaseHost != null && cfg.databasePasswordFile != null);
        message = "services.infernix.animationWorker requires either `databaseUrlFile` or both `databaseHost` and `databasePasswordFile`.";
      }
    ];

    services.infernix.gpuGate = {
      enable = true;
      inherit (cfg) device;
    };

    services.infernix.gpuAdmission = lib.mkIf cfg.enableGpuAdmission {
      enable = lib.mkDefault true;
      device = lib.mkDefault cfg.device;
    };

    services.infernix.workloadFabric =
      {
        enable = true;
        gpuDevice = cfg.device;
        inherit (cfg) concurrency memoryMax cpuQuota tasksMax;
        databaseUrlFile = effectiveDatabaseUrlFile;
        stagingRoot = "${cfg.modelRoot}/staging";
        cacheRoot = "${cfg.modelRoot}/cache";
        modelDirectories = [cfg.modelRoot];
        adapters =
          lib.mapAttrs (name: adapter: {
            workload = "animation";
            inherit (adapter) queues profiles;
            artifacts = true;
            command = "${cfg.modelRoot}/.venvs/${name}/bin/python";
            args = ["${cfg.adaptersRoot}/${adapter.script}"];
            # The venv's torch finds the NixOS GPU driver here.
            # nixpkgs' python needs the nix-ld library set for manylinux wheels.
            environment =
              {LD_LIBRARY_PATH = "${lib.makeLibraryPath config.programs.nix-ld.libraries}:/run/opengl-driver/lib";}
              // adapter.environment;
          })
          cfg.adapters;
      }
      // lib.optionalAttrs (cfg.hubTokenFile != null) {
        hubTokenFile = toString cfg.hubTokenFile;
      }
      // lib.optionalAttrs cfg.enableGpuAdmission {
        admission = {
          command = config.services.infernix.gpuAdmission.checkCommand;
          preempt = true;
        };
      };

    # Remote workers are not the table owner, so they must not run migrations.
    systemd.services.infernix-workload-migrate.enable = lib.mkIf (!cfg.runMigrations) (lib.mkForce false);

    # systemd reads EnvironmentFile before any ExecStartPre, so the DSN must
    # come from a unit ordered before both consumers. A tmpfs wipe on boot is
    # the reason this is a service, not a one-shot prefix.
    systemd.services.infernix-workload-database-env = lib.mkIf assembleDatabaseEnv {
      description = "Assemble the Infernix workload database DSN";
      before =
        ["infernix-workerd.service"]
        ++ lib.optional cfg.stagingJanitor.enable "infernix-staging-janitor.service";
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        ExecStart = lib.getExe databaseEnv;
      };
    };

    systemd.services.infernix-workerd = {
      requires = lib.mkForce (lib.optional assembleDatabaseEnv "infernix-workload-database-env.service");
      after = lib.mkForce (
        [
          "network-online.target"
          "infernix-animation-provision.service"
        ]
        ++ lib.optional assembleDatabaseEnv "infernix-workload-database-env.service"
      );
      # Keep animation workers independent of dark image-runner/Qwen provisioning.
      wants = lib.mkForce [
        "network-online.target"
        "infernix-animation-provision.service"
      ];
    };

    systemd.services.infernix-animation-provision = {
      description = "Provision animation adapter environments and checkpoints";
      wantedBy = ["multi-user.target"];
      after = ["network-online.target"];
      wants = ["network-online.target"];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "infernix-worker";
        Group = "infernix-worker";
        ExecStart = lib.getExe provision;
        StateDirectory = "infernix-workload";
        Restart = "on-failure";
        RestartSec = "5min";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ReadWritePaths = [cfg.modelRoot];
      };
    };

    systemd.services.infernix-qwen-image-provision = lib.mkIf (cfg.qwenProvisionProject != null) {
      description = "Provision dark Qwen streaming image runner Python environment";
      wantedBy = ["multi-user.target"];
      after = ["network-online.target"];
      wants = ["network-online.target"];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        TimeoutStartSec = "15min";
        User = "infernix-worker";
        Group = "infernix-worker";
        ExecStart = lib.getExe qwenProvision;
        StateDirectory = "infernix-workload";
        Restart = "on-failure";
        RestartSec = "5min";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ReadWritePaths = [cfg.modelRoot];
      };
    };

    systemd.services.infernix-staging-janitor = lib.mkIf cfg.stagingJanitor.enable {
      description = "Delete delivered or failed animation staging older than ${toString cfg.stagingJanitor.retentionDays} days";
      requires = lib.optional assembleDatabaseEnv "infernix-workload-database-env.service";
      wants = ["network-online.target"];
      after =
        ["network-online.target"]
        ++ lib.optional assembleDatabaseEnv "infernix-workload-database-env.service";
      serviceConfig = {
        Type = "oneshot";
        User = "infernix-worker";
        Group = "infernix-worker";
        EnvironmentFile = effectiveDatabaseUrlFile;
        ExecStart = lib.getExe janitor;
        Restart = "on-failure";
        RestartSec = "5min";
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ReadWritePaths = ["${cfg.modelRoot}/staging"];
      };
    };

    systemd.timers.infernix-staging-janitor = lib.mkIf cfg.stagingJanitor.enable {
      wantedBy = ["timers.target"];
      timerConfig = {
        OnCalendar = cfg.stagingJanitor.onCalendar;
        Persistent = true;
      };
    };

    systemd.tmpfiles.rules =
      ["d ${cfg.modelRoot} 0750 infernix-worker infernix-worker -"]
      ++ map (directory: "d ${cfg.modelRoot}/${directory} 0750 infernix-worker infernix-worker -")
      [
        ".hf"
        ".provisioned"
        ".uv-cache"
        ".venvs"
        "code"
        "frames"
        "weights"
      ]
      ++ lib.optional config.services.infernix.imageRunner.enable "z ${cfg.modelRoot}/staging 2770 infernix-worker infernix-image-handoff -";
  };
}
