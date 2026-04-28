# Computes goose customProviders and default settings from infernix endpoints
# as read-only options.
#
# This module declares options and exposes read-only outputs. It does NOT
# write to `programs.goose.*` itself — that would require a `programs.goose`
# module to be loaded for every user (which a consumer's home-manager
# sharedModules setup cannot guarantee). To get the auto-wiring without
# boilerplate, import `infernix.homeModules.goose` in users that also import
# a module declaring `programs.goose`; that module reads the outputs here and
# writes `programs.goose` directly.
#
# Per-endpoint translation:
#   - llama-swap → OpenAI-compatible customProvider
#   - ollama     → native goose ollama provider (no customProvider emitted)
#   - only local self-hosted Infernix endpoint types are handled here
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types filterAttrs foldlAttrs;
  cfg = config.services.infernix.goose;
  epCfg = config.services.infernix.endpoints;

  stripScheme = url:
    builtins.replaceStrings ["http://" "https://"] ["" ""] url;

  llamaSwapEndpoints = filterAttrs (_: ep: ep.type == "llama-swap") epCfg;

  mkLlamaSwapProvider = epName: ep: {
    name = epName;
    engine = "openai";
    display_name = epName;
    base_url = "${ep.url}/v1";
    models = lib.mapAttrsToList (_modelKey: model:
      {
        name = model.name;
      }
      // lib.optionalAttrs (model.ctxSize != null) {
        context_limit = model.ctxSize;
      })
    ep.models;
    supports_streaming = true;
    requires_auth = false;
  };

  generatedProviders =
    if cfg.enable
    then
      foldlAttrs (
        acc: epName: ep:
          acc // {${epName} = mkLlamaSwapProvider epName ep;}
      ) {}
      llamaSwapEndpoints
    else {};

  defaultEp =
    if cfg.defaultEndpoint != null
    then epCfg.${cfg.defaultEndpoint} or null
    else null;

  defaultModelEntry =
    if defaultEp != null && cfg.defaultModel != null
    then defaultEp.models.${cfg.defaultModel} or null
    else null;

  generatedSettings =
    if !cfg.enable || defaultEp == null || defaultModelEntry == null
    then {}
    else if defaultEp.type == "ollama"
    then {
      GOOSE_PROVIDER = "ollama";
      GOOSE_MODEL = defaultModelEntry.name;
      OLLAMA_HOST = stripScheme defaultEp.url;
    }
    else if defaultEp.type == "llama-swap"
    then {
      GOOSE_PROVIDER = cfg.defaultEndpoint;
      GOOSE_MODEL = defaultModelEntry.name;
    }
    else {};
in {
  options.services.infernix.goose = {
    enable = mkEnableOption "auto-generation of goose customProviders and default settings from infernix endpoints";

    defaultEndpoint = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "local-llama-swap";
      description = ''
        Name of the endpoint (from services.infernix.endpoints) to use as
        goose's default provider. Determines GOOSE_PROVIDER (and, for
        ollama endpoints, OLLAMA_HOST) in generatedSettings.

        Must reference a self-hosted endpoint type handled by goose through
        infernix, currently `ollama` or `llama-swap`.
      '';
    };

    defaultModel = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "coder";
      description = ''
        Attribute key of the model within `defaultEndpoint.models` to use
        as goose's default. Its `model.name` is written to GOOSE_MODEL.
      '';
    };

    generatedProviders = mkOption {
      type = types.attrs;
      readOnly = true;
      description = ''
        goose customProvider definitions auto-generated from every
        `llama-swap`-type entry in services.infernix.endpoints. Consumed
        by `infernix.homeModules.goose`; can also be wired manually into
        programs.goose.customProviders.

        ollama endpoints are not emitted here — they are represented
        through the native goose ollama provider in generatedSettings.
        Other endpoint types are skipped.
      '';
    };

    generatedSettings = mkOption {
      type = types.attrs;
      readOnly = true;
      description = ''
        Partial goose settings auto-generated from the chosen
        `defaultEndpoint` / `defaultModel`. Contains GOOSE_PROVIDER,
        GOOSE_MODEL, and (for ollama) OLLAMA_HOST.
      '';
    };
  };

  config = {
    services.infernix.goose.generatedProviders = generatedProviders;
    services.infernix.goose.generatedSettings = generatedSettings;

    assertions = lib.optionals cfg.enable [
      {
        assertion = cfg.defaultEndpoint == null || epCfg ? ${cfg.defaultEndpoint};
        message = "services.infernix.goose.defaultEndpoint refers to '${toString cfg.defaultEndpoint}' which is not declared in services.infernix.endpoints.";
      }
      {
        assertion =
          cfg.defaultEndpoint
          == null
          || builtins.elem ((epCfg.${cfg.defaultEndpoint} or {}).type or null) ["ollama" "llama-swap"];
        message = "services.infernix.goose.defaultEndpoint = '${toString cfg.defaultEndpoint}' is not a self-hosted goose endpoint — this module only handles ollama and llama-swap.";
      }
      {
        assertion =
          cfg.defaultModel
          == null
          || cfg.defaultEndpoint
          == null
          || (epCfg.${cfg.defaultEndpoint}.models or {}) ? ${cfg.defaultModel};
        message = "services.infernix.goose.defaultModel = '${toString cfg.defaultModel}' is not defined in endpoint '${toString cfg.defaultEndpoint}'.";
      }
    ];
  };
}
