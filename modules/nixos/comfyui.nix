{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.comfyui;
  gpuCfg = config.services.infernix.gpu;
  gpuLib = import ../../lib/gpu.nix {inherit lib;};

  defaultExtraModelPaths = pkgs.writeText "comfyui-extra-model-paths.yaml" ''
    canix:
      base_path: ${cfg.modelsDir}
      is_default: true
      checkpoints: checkpoints
      text_encoders: |
        text_encoders
        clip
      clip_vision: clip_vision
      configs: configs
      controlnet: controlnet
      diffusion_models: |
        diffusion_models
        unet
      embeddings: embeddings
      loras: loras
      upscale_models: upscale_models
      vae: vae
      audio_encoders: audio_encoders
      model_patches: model_patches
  '';

  extraModelPaths =
    if cfg.extraModelPathsConfig == null
    then defaultExtraModelPaths
    else cfg.extraModelPathsConfig;
in {
  options.services.infernix.comfyui = {
    enable = mkEnableOption "ComfyUI diffusion model interface";

    package = mkOption {
      type = types.nullOr types.package;
      default = null;
      description = ''
        ComfyUI package to run. Defaults to
        `services.infernix.gpu.pkgs.comfyui` when that attribute exists;
        otherwise it must be set explicitly.
      '';
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address ComfyUI listens on.";
    };

    port = mkOption {
      type = types.port;
      default = 8188;
      description = "Port ComfyUI listens on.";
    };

    firewallInterfaces = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Firewall interfaces where the ComfyUI TCP port is opened.";
    };

    stateDir = mkOption {
      type = types.str;
      default = "/var/lib/comfyui";
      description = "Base directory for ComfyUI runtime state.";
    };

    modelsDir = mkOption {
      type = types.str;
      default = "/var/lib/comfyui/models";
      description = "Base directory for ComfyUI model files.";
    };

    extraModelPathsConfig = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Optional extra_model_paths.yaml file. If unset, one is generated from modelsDir.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Environment variables for the ComfyUI service.";
    };
  };

  config = mkIf cfg.enable (let
    gpuOverrides = gpuLib.systemdGpuOverrides {
      vendor = gpuCfg.vendor;
      inherit (gpuCfg) visibleDevices;
      amd = gpuCfg.amd;
    };
    inferredPackage =
      if gpuCfg.pkgs != null && gpuCfg.pkgs ? comfyui
      then gpuCfg.pkgs.comfyui
      else null;
    package =
      if cfg.package != null
      then cfg.package
      else inferredPackage;
    serviceEnvironment =
      (gpuOverrides.Environment or [])
      ++ lib.mapAttrsToList (name: value: "${name}=${value}") cfg.environment;
  in {
    assertions = [
      {
        assertion = package != null;
        message = ''
          services.infernix.comfyui.package must be set unless
          services.infernix.gpu.pkgs provides a comfyui package.
        '';
      }
    ];

    users.groups.comfyui = {};

    users.users.comfyui = {
      isSystemUser = true;
      group = "comfyui";
      extraGroups = [
        "render"
        "video"
      ];
      home = cfg.stateDir;
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 0750 comfyui comfyui -"
      "d ${cfg.stateDir}/custom_nodes 0750 comfyui comfyui -"
      "d ${cfg.stateDir}/input 0750 comfyui comfyui -"
      "d ${cfg.stateDir}/output 0750 comfyui comfyui -"
      "d ${cfg.stateDir}/temp 0750 comfyui comfyui -"
      "d ${cfg.stateDir}/user 0750 comfyui comfyui -"
      "d ${cfg.modelsDir} 0775 comfyui users -"
      "d ${cfg.modelsDir}/audio_encoders 0775 comfyui users -"
      "d ${cfg.modelsDir}/checkpoints 0775 comfyui users -"
      "d ${cfg.modelsDir}/clip 0775 comfyui users -"
      "d ${cfg.modelsDir}/clip_vision 0775 comfyui users -"
      "d ${cfg.modelsDir}/configs 0775 comfyui users -"
      "d ${cfg.modelsDir}/controlnet 0775 comfyui users -"
      "d ${cfg.modelsDir}/diffusion_models 0775 comfyui users -"
      "d ${cfg.modelsDir}/embeddings 0775 comfyui users -"
      "d ${cfg.modelsDir}/loras 0775 comfyui users -"
      "d ${cfg.modelsDir}/model_patches 0775 comfyui users -"
      "d ${cfg.modelsDir}/text_encoders 0775 comfyui users -"
      "d ${cfg.modelsDir}/unet 0775 comfyui users -"
      "d ${cfg.modelsDir}/upscale_models 0775 comfyui users -"
      "d ${cfg.modelsDir}/vae 0775 comfyui users -"
    ];

    systemd.services.comfyui = {
      description = "ComfyUI diffusion model interface";
      after = ["network-online.target"];
      wants = ["network-online.target"];

      serviceConfig =
        gpuOverrides
        // {
          Environment = serviceEnvironment;
          User = "comfyui";
          Group = "comfyui";
          WorkingDirectory = cfg.stateDir;
          ExecStart = ''
            ${package}/bin/comfyui \
              --listen ${cfg.host} \
              --port ${toString cfg.port} \
              --base-directory ${cfg.stateDir} \
              --database-url sqlite:///${cfg.stateDir}/user/comfyui.db \
              --user-directory ${cfg.stateDir}/user \
              --input-directory ${cfg.stateDir}/input \
              --output-directory ${cfg.stateDir}/output \
              --temp-directory ${cfg.stateDir}/temp \
              --extra-model-paths-config ${extraModelPaths} \
              --disable-auto-launch \
              --log-stdout
          '';
          Restart = "on-failure";
          RestartSec = 5;
        };
    };

    networking.firewall.interfaces = lib.genAttrs cfg.firewallInterfaces (_: {
      allowedTCPPorts = [cfg.port];
    });
  });
}
