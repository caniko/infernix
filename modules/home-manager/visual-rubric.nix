# Computes visual-rubric configuration from infernix endpoints as read-only
# options and writes home.sessionVariables for the `configured` subcommand.
#
# Auto-discovery priority:
#   1. Explicit `vision.endpoint` / `vision.model` (highest priority).
#   2. Auto-discovery from osConfig.services.infernix.llama-swap.models
#      (available in NixOS-integrated mode — look for models with the "vlm"
#      alias which marks vision-language models).
#   3. Fallback: scan services.infernix.endpoints for llama-swap endpoints
#      (standalone HM mode).
#   4. Gives up with a clear eval error if nothing is found and enable = true.
#
# Unlike goose.nix and yeehaw.nix, this module writes home.sessionVariables
# directly rather than requiring an opt-in *-programs.nix module. That's
# safe because home.sessionVariables is a universally available HM option
# that causes no type errors for users who don't import it.
{
  config,
  lib,
  osConfig ? null,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types filterAttrs;
  cfg = config.services.infernix.visual-rubric;
  epCfg = config.services.infernix.endpoints;

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
    if cfg.enable && explicitVisionEp != null && explicitVisionModelEntry != null
    then {
      url = explicitVisionEp.url;
      model = explicitVisionModelEntry.name;
    }
    else null;

  # --- Auto-discovery from llama-swap (NixOS-integrated mode) ---

  llamaSwapCfg = osConfig.services.infernix.llama-swap or {};

  isVisionModel = model:
    builtins.elem "vlm" (model.aliases or [])
    || builtins.elem "captioner" (model.aliases or []);

  autoDiscoveredLlamaSwapModels =
    if osConfig != null && llamaSwapCfg.enable or false
    then filterAttrs (_: model: isVisionModel model) (llamaSwapCfg.models or {})
    else {};

  autoDiscoveredLlamaSwapVision =
    if cfg.vision.autoDiscover && autoDiscoveredLlamaSwapModels != {} then
      let
        firstKey = builtins.head (builtins.attrNames autoDiscoveredLlamaSwapModels);
        firstModel = autoDiscoveredLlamaSwapModels.${firstKey};
        port = toString llamaSwapCfg.port;
      in {
        url = "http://localhost:${port}";
        model = firstKey;
      }
    else null;

  # --- Auto-discovery from HM endpoints (standalone HM mode) ---

  llamaSwapEndpoints = filterAttrs (_: ep: ep.type == "llama-swap") epCfg;

  autoDiscoveredEndpointVision =
    if cfg.vision.autoDiscover && autoDiscoveredLlamaSwapVision == null then
      let
        # Find the first llama-swap endpoint with at least one model
        firstEp = builtins.head (builtins.attrNames llamaSwapEndpoints);
        firstEpConfig = llamaSwapEndpoints.${firstEp};
        firstModelKey = builtins.head (builtins.attrNames firstEpConfig.models);
        firstModelEntry = firstEpConfig.models.${firstModelKey};
      in {
        url = firstEpConfig.url;
        model = firstModelEntry.name;
      }
    else null;

  # --- Resolved vision config ---

  resolvedVision =
    explicitVision
    // (if explicitVision == null then (autoDiscoveredLlamaSwapVision // autoDiscoveredEndpointVision) else {});

  # --- ACP backend config ---

  rubricBackend = cfg.rubric.backend;

  rubricAcpArgs =
    if cfg.rubric.acpArgs != []
    then cfg.rubric.acpArgs
    else if rubricBackend == "codex-acp"
    then ["-c" "model=\"${cfg.rubric.modelOverride or "gpt-5.4-mini"}\"" "-c" "model_reasoning_effort=\"${cfg.rubric.effort or "medium"}\""]
    else ["acp"];

  rubricAcpArgsStr = builtins.concatStringsSep " " rubricAcpArgs;

  rubricBinary =
    if rubricBackend == "codex-acp"
    then "codex-acp"
    else "opencode";

  # --- Generated outputs ---

  generatedConfig =
    if cfg.enable && resolvedVision != null
    then {
      vision_url = resolvedVision.url;
      vision_model = resolvedVision.model;
      rubric_backend = rubricBackend;
      rubric_binary = rubricBinary;
      rubric_acp_args_str = rubricAcpArgsStr;
    }
    else {};
in {
  options.services.infernix.visual-rubric = {
    enable = mkEnableOption "auto-generation of visual-rubric config from infernix endpoints";

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
        type = types.enum ["opencode" "codex-acp"];
        default = "opencode";
        description = ''
          Which ACP backend to use for rubric scoring.
          - "opencode": uses the opencode binary with its configured model
            (e.g. DeepSeek V4). The model comes from opencode's config, not
            from infernix endpoints.
          - "codex-acp": uses codex-acp binary. The model is passed on the
            command line and must be specified in rubric.modelOverride.
        '';
      };

      acpArgs = mkOption {
        type = types.listOf types.str;
        default = [];
        example = ["acp"];
        description = ''
          Extra CLI arguments for the ACP binary.
          For opencode (default): ["acp"]
          For codex-acp: ["-c", "model=\"gpt-5.4-mini\"", "-c", "model_reasoning_effort=\"medium\""]
          When empty (default), the module derives appropriate args from
          the chosen backend.
        '';
      };

      modelOverride = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "deepseek-v4-flash";
        description = ''
          Model override for the rubric ACP backend. When backend is
          "opencode", this is ignored (model comes from opencode's config).
          When backend is "codex-acp", this value is used to construct the
          -c model="..." argument, falling back to a default if null.
        '';
      };

      effort = mkOption {
        type = types.nullOr (types.enum ["low" "medium" "high"]);
        default = null;
        description = ''
          Reasoning effort for the rubric ACP backend. Only meaningful
          for codex-acp; ignored for opencode.
        '';
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
        - rubric_backend: "opencode" or "codex-acp"
        - rubric_binary: path to the ACP binary
        - rubric_acp_args_str: space-separated ACP CLI arguments
      '';
    };
  };

  config = mkIf cfg.enable {
    services.infernix.visual-rubric.generatedConfig = generatedConfig;

    home.sessionVariables =
      if generatedConfig != {}
      then {
        VISUAL_RUBRIC_VISION_URL = generatedConfig.vision_url;
        VISUAL_RUBRIC_VISION_MODEL = generatedConfig.vision_model;
        VISUAL_RUBRIC_ACP_BINARY = generatedConfig.rubric_binary;
        VISUAL_RUBRIC_ACP_ARGS = generatedConfig.rubric_acp_args_str;
      }
      else {};

    assertions =
      [
        {
          assertion =
            !cfg.enable
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
            || (epCfg.${cfg.vision.endpoint}.models or {}) ? ${cfg.vision.model};
          message = "services.infernix.visual-rubric.vision.model = '${toString cfg.vision.model}' is not defined in endpoint '${toString cfg.vision.endpoint}'.";
        }
      ];
  };
}
