# NixOS module evaluation tests for comfyui.memoryGuard, gpuAdmission,
# animationWorker, and rhubarb-lip-sync packaging.
{
  pkgs,
  nixpkgs,
  nixosModule,
  self,
}: let
  inherit (pkgs) lib;
  system = pkgs.stdenv.hostPlatform.system;

  evalHost = extraModule:
    (nixpkgs.lib.nixosSystem {
      inherit system;
      modules = [
        nixosModule
        {
          system.stateVersion = "24.11";
          networking.hostName = "fixture-worker";
          services.infernix.gpu = {
            vendor = "cpu";
            inherit pkgs;
          };
        }
        extraModule
      ];
    }).config;

  # 1. ComfyUI memoryGuard disabled vs enabled
  comfyGuardOff = evalHost {
    services.infernix.comfyui = {
      enable = true;
      package = pkgs.hello;
    };
  };

  comfyGuardOn = evalHost {
    services.infernix.comfyui = {
      enable = true;
      package = pkgs.hello;
      memoryGuard = {
        enable = true;
        admitGiB = 50;
        admitPsi = 5;
        tripGiB = 4;
        tripPsi = 10;
        emergencyGiB = 4;
        watchdogSec = "3s";
        timeoutStartSec = 11;
        admissionCommand = ["/run/current-system/sw/bin/true" "--check"];
      };
    };
  };

  # 2. gpuAdmission: polling mode (onDemand = false) vs socket mode (onDemand = true)
  gpuAdmissionPolling = evalHost {
    services.infernix.gpuAdmission = {
      enable = true;
      package = pkgs.hello;
      onDemand = false;
      device = "arc-a770";
      backend = "fdinfo";
      pciAddress = "0000:0b:00.0";
      maxForeignVramGiB = 4.0;
      maxForeignBusyPercent = 25.0;
    };
  };

  gpuAdmissionOnDemand = evalHost {
    services.infernix.gpuAdmission = {
      enable = true;
      package = pkgs.hello;
      onDemand = true;
      timeoutSecs = 9;
      device = "worker-gpu1";
      backend = "fdinfo";
      maxForeignVramGiB = 3.5;
    };
  };

  # 3. animationWorker enabled with stubbed package/paths
  animationWorkerCfg = evalHost {
    services.infernix.gpuAdmission = {
      package = pkgs.hello;
      onDemand = true;
      maxForeignVramGiB = 4.0;
    };
    services.infernix.workloadFabric.package = pkgs.hello;
    services.infernix.animationWorker = {
      enable = true;
      device = "worker-gpu0";
      modelRoot = "/var/lib/animation-models";
      profilesDir = ../lib;
      databaseHost = "10.0.0.1";
      databasePasswordFile = "/run/secrets/db-pass";
      hubTokenFile = "/run/secrets/hub-token";
      qwenProvisionProject = "qwen-image-edit";
      adapters.sam2 = {
        project = "sam2";
        script = "sam2/adapter.py";
        queues = ["animation.segment"];
        profiles = ["sam2-fixture"];
      };
    };
  };

  rhubarbPkg = self.packages.${system}.rhubarb-lip-sync;
  helperGuardScript = self.lib.mkComfyuiMemoryGuard {
    inherit pkgs;
    admitGiB = 46;
    admitPsi = 5;
    tripGiB = 16;
    tripPsi = 10;
    emergencyGiB = 6;
  };
in
  # ComfyUI memoryGuard disabled
  assert !(comfyGuardOff.systemd.services ? comfyui-memory-guard);
  assert !(builtins.elem "comfyui-memory-guard.service" comfyGuardOff.systemd.services.comfyui.bindsTo);
  # ComfyUI memoryGuard enabled
  assert comfyGuardOn.systemd.services ? comfyui-memory-guard;
  assert comfyGuardOn.systemd.services.comfyui-memory-guard.serviceConfig.Type == "notify";
  assert comfyGuardOn.systemd.services.comfyui-memory-guard.serviceConfig.WatchdogSec == "3s";
  assert builtins.elem "comfyui-memory-guard.service" comfyGuardOn.systemd.services.comfyui.bindsTo;
  assert builtins.elem "comfyui-memory-guard.service" comfyGuardOn.systemd.services.comfyui.after;
  assert lib.hasInfix "--check" comfyGuardOn.systemd.services.comfyui-memory-guard.serviceConfig.ExecStart;
  # gpuAdmission polling mode
  assert gpuAdmissionPolling.systemd.services ? infernix-gpu-admission;
  assert !(gpuAdmissionPolling.systemd.services ? "infernix-gpu-admission@");
  assert !(gpuAdmissionPolling.systemd.sockets ? infernix-gpu-admission);
  assert gpuAdmissionPolling.systemd.services.infernix-gpu-admission.serviceConfig.Restart == "always";
  assert lib.hasInfix "--pci 0000:0b:00.0" gpuAdmissionPolling.systemd.services.infernix-gpu-admission.serviceConfig.ExecStart;
  assert builtins.elem "check" gpuAdmissionPolling.services.infernix.gpuAdmission.checkCommand;
  assert !(builtins.elem "--socket" gpuAdmissionPolling.services.infernix.gpuAdmission.checkCommand);
  # gpuAdmission onDemand mode
  assert gpuAdmissionOnDemand.systemd.sockets ? infernix-gpu-admission;
  assert gpuAdmissionOnDemand.systemd.services ? "infernix-gpu-admission@";
  assert !(gpuAdmissionOnDemand.systemd.services ? infernix-gpu-admission);
  assert gpuAdmissionOnDemand.systemd.sockets.infernix-gpu-admission.socketConfig.ListenStream == "/run/infernix-gpu-admission/worker-gpu1.sock";
  assert gpuAdmissionOnDemand.systemd.services."infernix-gpu-admission@".serviceConfig.Restart == "no";
  assert lib.hasInfix "--json" gpuAdmissionOnDemand.systemd.services."infernix-gpu-admission@".serviceConfig.ExecStart;
  assert lib.hasInfix "--once" gpuAdmissionOnDemand.systemd.services."infernix-gpu-admission@".serviceConfig.ExecStart;
  assert builtins.elem "--socket" gpuAdmissionOnDemand.services.infernix.gpuAdmission.checkCommand;
  assert builtins.elem "/run/infernix-gpu-admission/worker-gpu1.sock" gpuAdmissionOnDemand.services.infernix.gpuAdmission.checkCommand;
  # animationWorker
  assert animationWorkerCfg.services.infernix.gpuAdmission.enable;
  assert animationWorkerCfg.services.infernix.gpuAdmission.device == "worker-gpu0";
  assert animationWorkerCfg.services.infernix.gpuGate.enable;
  assert animationWorkerCfg.services.infernix.gpuGate.device == "worker-gpu0";
  assert animationWorkerCfg.services.infernix.workloadFabric.admission.command == animationWorkerCfg.services.infernix.gpuAdmission.checkCommand;
  assert animationWorkerCfg.services.infernix.workloadFabric.admission.preempt;
  assert animationWorkerCfg.systemd.services.infernix-workload-migrate.enable == false;
  assert animationWorkerCfg.systemd.services ? infernix-workload-database-env;
  assert animationWorkerCfg.systemd.services ? infernix-animation-provision;
  assert animationWorkerCfg.systemd.services ? infernix-qwen-image-provision;
  assert animationWorkerCfg.systemd.services ? infernix-staging-janitor;
  assert animationWorkerCfg.systemd.timers ? infernix-staging-janitor;
  assert builtins.elem "infernix-workload-database-env.service" animationWorkerCfg.systemd.services.infernix-workerd.requires;
  assert builtins.elem "infernix-animation-provision.service" animationWorkerCfg.systemd.services.infernix-workerd.after;
  # rhubarb-lip-sync + lib exports
  assert rhubarbPkg.pname == "rhubarb-lip-sync";
  assert rhubarbPkg.version == "1.14.0";
  assert rhubarbPkg.meta.mainProgram == "rhubarb";
  assert builtins.isString rhubarbPkg.drvPath;
  assert builtins.pathExists self.lib.animationPrefetchScript;
    pkgs.runCommand "infernix-runtime-modules-check" {} ''
      test -x ${helperGuardScript}
      grep -Fq 'prefetch_profile' ${self.lib.animationPrefetchScript}
      grep -Fq 'prefetch_profile sam2-fixture' ${animationWorkerCfg.systemd.services.infernix-animation-provision.serviceConfig.ExecStart}
      touch "$out"
    ''
