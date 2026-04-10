{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit
    (lib)
    mkEnableOption
    mkOption
    mkIf
    types
    mapAttrsToList
    concatStringsSep
    getExe'
    getExe
    optional
    flatten
    ;
  gpuCfg = config.services.infernis.gpu;
  cfg = config.services.infernis.llama-swap;
  gpuLib = import ../../lib/gpu.nix {inherit lib;};

  llama-cpp = gpuLib.overrideLlamaCpp {
    vendor = gpuCfg.vendor;
    pkgs = gpuCfg.pkgs;
    amd = gpuCfg.amd;
    extraCmakeFlags = cfg.llamaCpp.extraCmakeFlags;
    flashAttention = cfg.llamaCpp.flashAttention;
  };
  llama-server = getExe' llama-cpp "llama-server";

  commonArgs = [
    "--n-gpu-layers 99"
    "--flash-attn on"
    "--cache-type-k q8_0"
    "--cache-type-v q4_0"
    "--threads -1"
    "--jinja"
    "--no-context-shift"
    "--no-webui"
  ];

  mkModelCmd = _name: model: let
    draftArgs =
      optional (model.draft != null) "-md ${cfg.modelsDir}/${model.draft.file}"
      ++ optional (model.draft != null) "--draft-max ${toString model.draft.draftMax}"
      ++ optional (model.draft != null) "--draft-min ${toString model.draft.draftMin}"
      ++ optional (model.draft != null) "-ngld ${toString model.draft.nGpuLayers}";
  in
    concatStringsSep " " ([
        llama-server
        "--port \${PORT}"
        "-m ${cfg.modelsDir}/${model.file}"
      ]
      ++ draftArgs
      ++ [
        "--ctx-size ${toString model.ctxSize}"
      ]
      ++ commonArgs
      ++ model.extraArgs);

  # Collect all files that need downloading (main models + draft models)
  downloadFiles = flatten (mapAttrsToList (_name: model:
    [{inherit (model) repo file;}]
    ++ optional (model.draft != null) {inherit (model.draft) repo file;})
  cfg.models);

  draftSubmodule = types.submodule {
    options = {
      file = mkOption {
        type = types.str;
        description = "GGUF filename for the draft model.";
      };

      repo = mkOption {
        type = types.str;
        description = "HuggingFace repo for the draft model.";
      };

      draftMax = mkOption {
        type = types.int;
        default = 16;
        description = "Maximum speculative decoding tokens.";
      };

      draftMin = mkOption {
        type = types.int;
        default = 2;
        description = "Minimum speculative decoding tokens.";
      };

      nGpuLayers = mkOption {
        type = types.int;
        default = 99;
        description = "GPU layers for draft model.";
      };
    };
  };

  modelSubmodule = types.submodule {
    options = {
      file = mkOption {
        type = types.str;
        description = "GGUF filename.";
      };

      repo = mkOption {
        type = types.str;
        description = "HuggingFace repo (e.g. unsloth/Qwen3-Coder-Next-GGUF).";
      };

      ctxSize = mkOption {
        type = types.int;
        description = "Context window size.";
      };

      aliases = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Model aliases for the llama-swap API.";
      };

      ttl = mkOption {
        type = types.int;
        default = 300;
        description = "Time-to-live in seconds before model is unloaded.";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional CLI args appended to the llama-server command.";
      };

      draft = mkOption {
        type = types.nullOr draftSubmodule;
        default = null;
        description = "Optional speculative decoding draft model.";
      };
    };
  };
in {
  options.services.infernis.llama-swap = {
    enable = mkEnableOption "llama-swap model orchestrator";

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Address to bind llama-swap to.";
    };

    port = mkOption {
      type = types.port;
      default = 8013;
      description = "llama-swap API port.";
    };

    modelsDir = mkOption {
      type = types.path;
      description = "Directory where GGUF model files are stored.";
    };

    hfTokenPath = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to file containing HuggingFace API token for model downloads.";
    };

    logLevel = mkOption {
      type = types.str;
      default = "info";
      description = "llama-swap log level.";
    };

    healthCheckTimeout = mkOption {
      type = types.int;
      default = 120;
      description = "Health check timeout in seconds.";
    };

    llamaCpp = {
      extraCmakeFlags = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional CMake flags for the llama-cpp build.";
      };

      flashAttention = {
        allQuants = mkOption {
          type = types.bool;
          default = false;
          description = "Enable GGML_HIP_FA_ALL_QUANTS for flash attention on all KV cache quant types (AMD only).";
        };
      };
    };

    models = mkOption {
      type = types.attrsOf modelSubmodule;
      default = {};
      description = "Model definitions for llama-swap.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to open firewall ports for llama-swap.";
    };
  };

  config = mkIf cfg.enable {
    services.llama-swap = {
      enable = true;
      listenAddress = cfg.host;
      port = cfg.port;

      settings = {
        logLevel = cfg.logLevel;
        healthCheckTimeout = cfg.healthCheckTimeout;

        models =
          lib.mapAttrs (name: model: {
            cmd = mkModelCmd name model;
            inherit (model) ttl aliases;
          })
          cfg.models;
      };
    };

    # GPU-specific systemd overrides
    systemd.services.llama-swap.serviceConfig =
      (gpuLib.systemdGpuOverrides {
        vendor = gpuCfg.vendor;
        inherit (gpuCfg) visibleDevices;
        amd = gpuCfg.amd;
      })
      // {
        ReadOnlyPaths = [cfg.modelsDir];
      };

    # Model download service
    systemd.services.infernis-download = let
      curl = getExe pkgs.curl;
      authHeader =
        if cfg.hfTokenPath != null
        then ''-H "Authorization: Bearer $(cat ${cfg.hfTokenPath})"''
        else "";
      hfDownload = {
        repo,
        file,
      }: ''
        if [ ! -f "${cfg.modelsDir}/${file}" ]; then
          echo "Downloading ${file} from ${repo}"
          ${curl} -L --fail --progress-bar \
            ${authHeader} \
            -o "${cfg.modelsDir}/${file}" \
            "https://huggingface.co/${repo}/resolve/main/${file}"
        fi
      '';
    in {
      description = "Download GGUF models for infernis llama-swap";
      wants = ["network-online.target"];
      after = ["network-online.target"];
      wantedBy = ["llama-swap.service"];
      before = ["llama-swap.service"];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
      };
      script = ''
        mkdir -p "${cfg.modelsDir}"
        ${concatStringsSep "\n" (map hfDownload downloadFiles)}
      '';
    };

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [cfg.port];
  };
}
