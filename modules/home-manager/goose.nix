# Computes goose customProviders and default settings from infernis endpoints
# as read-only options.
#
# This module declares options and exposes read-only outputs. It does NOT
# write to `programs.goose.*` itself — that would require goose-hm to be
# loaded for every user (which canix's home-manager sharedModules setup
# cannot guarantee). To get the auto-wiring without boilerplate, import
# `infernis.homeModules.goose` in users that also import goose-hm; that
# module reads the outputs here and writes `programs.goose` directly.
#
# Per-endpoint translation:
#   - llama-swap → OpenAI-compatible customProvider
#   - ollama     → native goose ollama provider (no customProvider emitted)
#   - claude     → skipped entirely
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types filterAttrs foldlAttrs;
  cfg = config.services.infernis.goose;
  epCfg = config.services.infernis.endpoints;

  stripScheme = url:
    builtins.replaceStrings ["http://" "https://"] ["" ""] url;

  llamaSwapEndpoints = filterAttrs (_: ep: ep.type == "llama-swap") epCfg;

  mkLlamaSwapProvider = epName: ep: {
    name = epName;
    engine = "openai";
    display_name = epName;
    base_url = "${ep.url}/v1/chat/completions";
    api_key_env = "INFERNIS_${lib.toUpper (builtins.replaceStrings ["-"] ["_"] epName)}_KEY";
    models =
      lib.mapAttrsToList (_modelKey: model:
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
  options.services.infernis.goose = {
    enable = mkEnableOption "auto-generation of goose customProviders and default settings from infernis endpoints";

    defaultEndpoint = mkOption {
      type = types.nullOr types.str;
      default = null;
      example = "local-llama-swap";
      description = ''
        Name of the endpoint (from services.infernis.endpoints) to use as
        goose's default provider. Determines GOOSE_PROVIDER (and, for
        ollama endpoints, OLLAMA_HOST) in generatedSettings.

        Must not reference a `claude`-type endpoint — those are skipped
        by this module entirely.
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
        `llama-swap`-type entry in services.infernis.endpoints. Consumed
        by `infernis.homeModules.goose`; can also be wired manually into
        programs.goose.customProviders.

        ollama endpoints are not emitted here — they are represented
        through the native goose ollama provider in generatedSettings.
        claude endpoints are skipped.
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
    services.infernis.goose.generatedProviders = generatedProviders;
    services.infernis.goose.generatedSettings = generatedSettings;

    assertions = lib.optionals cfg.enable [
      {
        assertion = cfg.defaultEndpoint == null || epCfg ? ${cfg.defaultEndpoint};
        message = "services.infernis.goose.defaultEndpoint refers to '${toString cfg.defaultEndpoint}' which is not declared in services.infernis.endpoints.";
      }
      {
        assertion =
          cfg.defaultEndpoint
          == null
          || ((epCfg.${cfg.defaultEndpoint} or {}).type or null) != "claude";
        message = "services.infernis.goose.defaultEndpoint = '${toString cfg.defaultEndpoint}' is a claude endpoint — this module only handles ollama and llama-swap.";
      }
      {
        assertion =
          cfg.defaultModel
          == null
          || cfg.defaultEndpoint
          == null
          || (epCfg.${cfg.defaultEndpoint}.models or {}) ? ${cfg.defaultModel};
        message = "services.infernis.goose.defaultModel = '${toString cfg.defaultModel}' is not defined in endpoint '${toString cfg.defaultEndpoint}'.";
      }
    ];
  };
}
