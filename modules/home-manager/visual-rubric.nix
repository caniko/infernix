# Computes visual-rubric configuration from infernix endpoints as read-only
# options and writes a TOML config file for the `configured` subcommand.
#
# Auto-discovery priority:
#   1. Explicit `vision.endpoint` / `vision.model` (highest priority).
#   2. Auto-discovery from osConfig.services.infernix.llama-swap.models
#      (available in NixOS-integrated mode — look for models with the "vlm"
#      alias which marks vision-language models).
#   3. Fallback: scan services.infernix.endpoints for llama-swap endpoints
#      (standalone HM mode).
#   4. Gives up with a clear eval error if nothing is found and enable = true.
{ config
, lib
, pkgs
, osConfig ? null
, infernixVisualRubric ? null
, ...
}:
let
  inherit (lib) mkEnableOption mkOption mkIf types filterAttrs optionalAttrs;
  cfg = config.services.infernix.visual-rubric;
  epCfg = config.services.infernix.endpoints;
  system = pkgs.stdenv.hostPlatform.system;
  isPipeline = cfg.mode == "pipeline";

  # --- Explicit endpoint resolution ---

  explicitVisionEp =
    if cfg.vision.endpoint != null
    then epCfg.${cfg.vision.endpoint} or null
    else null;

  explicitVisionModelEntry =
    if explicitVisionEp != null && cfg.vision.model != null
    then explicitVisionEp.models.${cfg.vision.model} or null
    else null;

  explicitVision =
    if cfg.enable && isPipeline && explicitVisionEp != null && explicitVisionModelEntry != null
    then {
      url = explicitVisionEp.url;
      model = explicitVisionModelEntry.name;
    }
    else null;

  # --- Auto-discovery from llama-swap (NixOS-integrated mode) ---

  llamaSwapCfg = osConfig.services.infernix.llama-swap or { };

  isVisionModel = model:
    builtins.elem "vlm" (model.aliases or [ ])
    || builtins.elem "captioner" (model.aliases or [ ]);

  autoDiscoveredLlamaSwapModels =
    if osConfig != null && llamaSwapCfg.enable or false
    then filterAttrs (_: model: isVisionModel model) (llamaSwapCfg.models or { })
    else { };

  autoDiscoveredLlamaSwapVision =
    if isPipeline && cfg.vision.autoDiscover && autoDiscoveredLlamaSwapModels != { } then
      let
        firstKey = builtins.head (builtins.attrNames autoDiscoveredLlamaSwapModels);
        firstModel = autoDiscoveredLlamaSwapModels.${firstKey};
        port = toString llamaSwapCfg.port;
      in
      {
        url = "http://localhost:${port}";
        model = firstKey;
      }
    else null;

  # --- Auto-discovery from HM endpoints (standalone HM mode) ---

  llamaSwapEndpoints = filterAttrs (_: ep: ep.type == "llama-swap") epCfg;

  autoDiscoveredEndpointVision =
    if isPipeline
      && cfg.vision.autoDiscover
      && autoDiscoveredLlamaSwapVision == null
      && llamaSwapEndpoints != { }
    then
      let
        # Find the first llama-swap endpoint with at least one model
        firstEp = builtins.head (builtins.attrNames llamaSwapEndpoints);
        firstEpConfig = llamaSwapEndpoints.${firstEp};
        firstModelKey = builtins.head (builtins.attrNames firstEpConfig.models);
        firstModelEntry = firstEpConfig.models.${firstModelKey};
      in
      {
        url = firstEpConfig.url;
        model = firstModelEntry.name;
      }
    else null;

  # --- Resolved vision config ---

  resolvedVision =
    if explicitVision != null then explicitVision
    else if autoDiscoveredLlamaSwapVision != null then autoDiscoveredLlamaSwapVision
    else autoDiscoveredEndpointVision;

  # --- ACP backend config ---

  rubricBackend =
    if cfg.rubric.backend != null
    then cfg.rubric.backend
    else if cfg.mode == "direct"
    then "codex-acp"
    else "opencode";

  rubricModel =
    if cfg.rubric.modelOverride != null
    then cfg.rubric.modelOverride
    else if rubricBackend == "codex-acp"
    then "gpt-5.5"
    else null;

  rubricEffort =
    if cfg.rubric.effort != null
    then cfg.rubric.effort
    else if rubricBackend == "codex-acp"
    then "medium"
    else null;

  rubricAcpArgs =
    if cfg.rubric.acpArgs != [ ]
    then cfg.rubric.acpArgs
    else if rubricBackend == "codex-acp"
    then [ "-c" "model=\"${rubricModel}\"" "-c" "model_reasoning_effort=\"${rubricEffort}\"" ]
    else [ "acp" ];

  rubricAcpArgsStr = builtins.concatStringsSep " " rubricAcpArgs;

  rubricBinary =
    if rubricBackend == "codex-acp"
    then "codex-acp"
    else "opencode";

  # --- Package resolution ---

  visualRubricPackages =
    if infernixVisualRubric != null
      && builtins.hasAttr "packages" infernixVisualRubric
      && builtins.hasAttr system infernixVisualRubric.packages
    then infernixVisualRubric.packages.${system}
    else { };

  visualRubricDefaultPackage = visualRubricPackages.default or null;
  visualRubricCodexAcpPackage = visualRubricPackages."codex-acp" or null;

  resolvedPackage =
    if cfg.package != null
    then cfg.package
    else if rubricBackend == "codex-acp"
    then visualRubricCodexAcpPackage
    else visualRubricDefaultPackage;

  expectedPackageAttr =
    if rubricBackend == "codex-acp"
    then "codex-acp"
    else "default";

  # --- Generated outputs ---

  generatedConfig =
    if cfg.enable && (!isPipeline || resolvedVision != null)
    then
      {
        mode = cfg.mode;
        rubric_backend = rubricBackend;
        rubric_binary = rubricBinary;
        rubric_acp_args = rubricAcpArgs;
        rubric_acp_args_str = rubricAcpArgsStr;
        sequence_max_frames = cfg.sequence.maxFrames;
        sequence_require_transition = cfg.sequence.requireTransition;
      }
      // optionalAttrs (rubricModel != null) {
        rubric_model = rubricModel;
      }
      // optionalAttrs (rubricEffort != null) {
        rubric_effort = rubricEffort;
      }
      // optionalAttrs isPipeline {
        vision_url = resolvedVision.url;
        vision_model = resolvedVision.model;
      }
    else { };

  generatedToml =
    {
      mode = generatedConfig.mode;
      sequence = {
        max_frames = generatedConfig.sequence_max_frames;
        require_transition = generatedConfig.sequence_require_transition;
      };
      rubric =
        {
          backend = generatedConfig.rubric_backend;
        }
        // optionalAttrs (generatedConfig ? rubric_model) {
          model = generatedConfig.rubric_model;
        }
        // optionalAttrs (generatedConfig ? rubric_effort) {
          effort = generatedConfig.rubric_effort;
        }
        // optionalAttrs (generatedConfig.mode == "pipeline") {
          args = generatedConfig.rubric_acp_args;
        };
    }
    // optionalAttrs (generatedConfig.mode == "pipeline") {
      vision = {
        url = generatedConfig.vision_url;
        model = generatedConfig.vision_model;
      };
    };
in
{
  options.services.infernix.visual-rubric = {
    enable = mkEnableOption "auto-generation of visual-rubric config from infernix endpoints";

    mode = mkOption {
      type = types.enum [ "direct" "pipeline" ];
      default = "direct";
      description = ''
        visual-rubric backend mode. "direct" sends one multimodal prompt to
        codex-acp. "pipeline" uses a vision endpoint first, then an ACP
        rubric scorer.
      '';
    };

    package = mkOption {
      type = types.nullOr types.package;
      default = null;
      description = ''
        visual-rubric package to install. When null, the module selects the
        package from infernix's visual-rubric input: codex-acp backends use
        packages.''${system}.codex-acp, and other backends use
        packages.''${system}.default.
      '';
    };

    vision = {
      autoDiscover = mkOption {
        type = types.bool;
        default = true;
        description = ''
          Automatically discover the vision endpoint from the host's
          llama-swap or HM endpoint configuration when
          vision.endpoint is not explicitly set.
        '';
      };

      endpoint = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "local-llama-swap";
        description = ''
          Name of the endpoint (from services.infernix.endpoints) hosting
          the vision model. When null and autoDiscover is true, the module
          scans llama-swap models (NixOS-integrated) or HM endpoints
          (standalone HM) for a suitable vision model.
        '';
      };

      model = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "vlm";
        description = ''
          Attribute key of the model within the endpoint to use for vision
          extraction. Its `model.name` is resolved to the actual model
          identifier passed to the API.
        '';
      };
    };

    rubric = {
      backend = mkOption {
        type = types.nullOr (types.enum [ "opencode" "codex-acp" ]);
        default = null;
        description = ''
          Which ACP backend to use for rubric scoring.
          When null, direct mode defaults to "codex-acp" and pipeline mode
          defaults to "opencode".
          - "opencode": uses the opencode binary with its configured model
            (e.g. DeepSeek V4). The model comes from opencode's config, not
            from infernix endpoints.
          - "codex-acp": uses the subscription-backed codex-acp binary.
        '';
      };

      acpArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        example = [ "acp" ];
        description = ''
          Extra CLI arguments for the ACP binary.
          For opencode (default): ["acp"]
          For codex-acp: ["-c", "model=\"gpt-5.5\"", "-c", "model_reasoning_effort=\"medium\""]
          When empty (default), the module derives appropriate args from
          the chosen backend.
        '';
      };

      modelOverride = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "deepseek-v4-flash";
        description = ''
          Model override for the rubric ACP backend. Direct codex-acp mode
          defaults to gpt-5.5 when this is null.
        '';
      };

      effort = mkOption {
        type = types.nullOr (types.enum [ "low" "medium" "high" ]);
        default = null;
        description = ''
          Reasoning effort for the rubric ACP backend. Only meaningful
          for codex-acp; ignored for opencode.
        '';
      };
    };

    sequence = {
      maxFrames = mkOption {
        type = types.ints.positive;
        default = 8;
        description = "Maximum ordered screenshot checkpoints accepted by visual-rubric.";
      };

      requireTransition = mkOption {
        type = types.bool;
        default = true;
        description = "Require sequence rubrics to assess before/after transition semantics.";
      };
    };

    generatedConfig = mkOption {
      type = types.attrs;
      readOnly = true;
      description = ''
        Resolved visual-rubric configuration generated from auto-discovery
        or explicit endpoint configuration. Contains:
        - vision_url: base URL of the vision API
        - vision_model: model name for the vision API
        - mode: "direct" or "pipeline"
        - rubric_backend: "opencode" or "codex-acp"
        - rubric_binary: path to the ACP binary
        - rubric_acp_args: ACP CLI argument list
        - rubric_acp_args_str: space-separated ACP CLI arguments
        - sequence_max_frames: maximum ordered checkpoints
        - sequence_require_transition: require before/after semantics
      '';
    };
  };

  config = mkIf cfg.enable {
    services.infernix.visual-rubric.generatedConfig = generatedConfig;

    home.packages = mkIf (cfg.enable && resolvedPackage != null) [
      resolvedPackage
    ];

    xdg.configFile."visual-rubric/config.toml" =
      if generatedConfig != { }
      then {
        source = (pkgs.formats.toml { }).generate "visual-rubric-config" generatedToml;
      }
      else { };

    assertions =
      [
        {
          assertion =
            !cfg.enable
            || resolvedPackage != null;
          message = ''
            services.infernix.visual-rubric could not resolve a visual-rubric
            package for ${system}. Expected infernix's visual-rubric input to
            expose packages.${system}.${expectedPackageAttr}, or set
            services.infernix.visual-rubric.package explicitly.
          '';
        }
        {
          assertion =
            !cfg.enable
            || !isPipeline
            || resolvedVision != null
            || cfg.vision.endpoint != null;
          message = ''
            services.infernix.visual-rubric is enabled but no vision model
            could be found. Either set vision.endpoint / vision.model
            explicitly, or enable autoDiscover (default) so the module can
            scan llama-swap or endpoint models.
          '';
        }
      ]
      ++ lib.optionals (cfg.vision.endpoint != null) [
        {
          assertion = epCfg ? ${cfg.vision.endpoint};
          message = "services.infernix.visual-rubric.vision.endpoint refers to '${toString cfg.vision.endpoint}' which is not declared in services.infernix.endpoints.";
        }
        {
          assertion =
            cfg.vision.model
            == null
            || (epCfg.${cfg.vision.endpoint}.models or { }) ? ${cfg.vision.model};
          message = "services.infernix.visual-rubric.vision.model = '${toString cfg.vision.model}' is not defined in endpoint '${toString cfg.vision.endpoint}'.";
        }
      ];
  };
}
