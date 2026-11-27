# GPU admission for workload workers on shared desktop PCs: a root probe
# measures foreign VRAM residency and engine time against calibrated limits and
# publishes one verdict per device; `infernix-gpu-admission check` is the
# unprivileged worker contract (exit 0 admits, 75 denies) and `override` is the
# operator escape hatch.
{
  config,
  infernixSelf,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.infernix.gpuAdmission;
  # `toString` renders 4.0 as "4.000000"; JSON keeps the shortest exact form.
  number = builtins.toJSON;
  socketPath = "${cfg.stateDir}/${cfg.device}.sock";
  checkCommand =
    [
      (lib.getExe cfg.package)
      "check"
      "--device"
      cfg.device
      "--state-dir"
      cfg.stateDir
    ]
    ++ lib.optionals cfg.onDemand ["--socket" socketPath "--timeout-secs" (toString cfg.timeoutSecs)];
in {
  options.services.infernix.gpuAdmission = {
    enable = lib.mkEnableOption "GPU admission probe for workload workers on this desktop";

    package = lib.mkOption {
      type = lib.types.package;
      default = infernixSelf.packages.${pkgs.stdenv.hostPlatform.system}.infernix-gpu-admission;
      defaultText = lib.literalExpression "inputs.infernix.packages.<system>.infernix-gpu-admission";
      description = "Package providing the `infernix-gpu-admission` binary.";
    };

    onDemand = lib.mkEnableOption "socket-activated GPU measurement instead of idle polling";

    timeoutSecs = lib.mkOption {
      type = lib.types.ints.between 1 300;
      default = 8;
      description = "Demand measurement deadline; must fit inside the worker's admission timeout.";
    };

    device = lib.mkOption {
      type = lib.types.str;
      description = "Device label used for the verdict and override file names.";
      example = "arc-a770";
    };

    backend = lib.mkOption {
      type = lib.types.enum ["fdinfo" "nvidia-smi"];
      default = "fdinfo";
      description = ''
        Measurement backend. `fdinfo` reads DRM clients from
        `/proc/<pid>/fdinfo` (amdgpu, i915, xe); `nvidia-smi` is for the
        proprietary NVIDIA driver, which exposes no DRM fdinfo.
      '';
    };

    pciAddress = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Only count `fdinfo` clients on this PCI address.";
      example = "0000:0b:00.0";
    };

    nvidiaIndex = lib.mkOption {
      type = lib.types.int;
      default = 0;
      description = ''
        Index of the GPU the `nvidia-smi` backend measures, as listed by
        `nvidia-smi -L`. Ignored by the `fdinfo` backend.
      '';
    };

    excludeUsers = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = ["infernix-worker"];
      description = "Users whose own GPU use never blocks the worker.";
    };

    maxForeignVramGiB = lib.mkOption {
      type = lib.types.float;
      description = ''
        Deny while another user's VRAM residency exceeds this many GiB.
        Deliberately has no default: each GPU and gaming profile must be
        calibrated on its host.
      '';
      example = 4.0;
    };

    maxForeignBusyPercent = lib.mkOption {
      type = lib.types.float;
      default = 25.0;
      description = "Deny while another user's busiest engine exceeds this percent of wall time.";
    };

    intervalSecs = lib.mkOption {
      type = lib.types.int;
      default = 10;
      description = "Seconds between published verdicts.";
    };

    stateDir = lib.mkOption {
      type = lib.types.str;
      default = "/run/infernix-gpu-admission";
      description = "Directory holding the verdict and override files.";
    };

    checkCommand = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      readOnly = true;
      description = ''
        Full argv a worker runs before claiming a GPU job and while one runs.
        The first stdout line is the reason; exit 0 admits and 75 denies.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    services.infernix.gpuAdmission.checkCommand = checkCommand;

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 0755 root root -"
    ];

    systemd.sockets.infernix-gpu-admission = lib.mkIf cfg.onDemand {
      description = "On-demand GPU admission for ${cfg.device}";
      wantedBy = ["sockets.target"];
      socketConfig = {
        ListenStream = socketPath;
        Accept = true;
        # The verdict is already public. Requests carry no arguments or commands.
        SocketMode = "0666";
        DirectoryMode = "0755";
        MaxConnections = 2;
        TriggerLimitIntervalSec = "1min";
        TriggerLimitBurst = 60;
        RemoveOnStop = true;
      };
    };

    systemd.services.${
      if cfg.onDemand
      then "infernix-gpu-admission@"
      else "infernix-gpu-admission"
    } = {
      description = "GPU admission probe for ${cfg.device}";
      wantedBy = lib.optional (!cfg.onDemand) "multi-user.target";
      # nvidia-smi is a driver tool, not a runtime input; fdinfo needs none.
      path =
        [pkgs.coreutils]
        ++ lib.optionals (cfg.backend == "nvidia-smi") [config.hardware.nvidia.package.bin];
      serviceConfig =
        {
          Type = "simple";
          ExecStart = lib.escapeShellArgs (
            [
              (lib.getExe cfg.package)
            ]
            ++ lib.optional cfg.onDemand "--json"
            ++ [
              "probe"
              "--device"
              cfg.device
              "--backend"
              cfg.backend
              "--max-foreign-vram-gib"
              (number cfg.maxForeignVramGiB)
              "--max-foreign-busy-percent"
              (number cfg.maxForeignBusyPercent)
              "--interval-secs"
              (toString cfg.intervalSecs)
              "--state-dir"
              cfg.stateDir
            ]
            ++ lib.optional cfg.onDemand "--once"
            ++ lib.optionals (cfg.pciAddress != null) ["--pci" cfg.pciAddress]
            ++ lib.optionals (cfg.backend == "nvidia-smi") [
              "--nvidia-index"
              (toString cfg.nvidiaIndex)
            ]
            ++ lib.concatMap (user: ["--exclude-user" user]) cfg.excludeUsers
          );
          Restart =
            if cfg.onDemand
            then "no"
            else "always";
          RestartSec = "5s";
          Nice = 10;
          # Reading other users' /proc needs root, so the unit stays privileged
          # and is contained by the filesystem view instead.
          # ProtectProc must stay at its default: `invisible` would hide other
          # users' /proc, and this probe exists to measure exactly that.
          ProtectSystem = "strict";
          ReadWritePaths = [cfg.stateDir];
          PrivateTmp = true;
          NoNewPrivileges = true;
        }
        // lib.optionalAttrs cfg.onDemand {
          StandardInput = "socket";
          StandardOutput = "socket";
          StandardError = "journal";
          RuntimeMaxSec = cfg.timeoutSecs + 2;
          MemoryMax = "512M";
          MemorySwapMax = 0;
          TasksMax = 64;
        };
    };
  };
}
