# Computes yeeHaw steeds from infernix endpoints as a read-only option.
#
# Because home-manager sharedModules apply to all users — but yeeHaw may
# only be imported for some of them — this module does NOT write to
# `programs.yh.steeds` itself (that would fail type-checking for users
# without yeeHaw's HM module, since mkIf still registers the option path).
#
# Instead, it exposes `services.infernix.yeehaw.generatedSteeds`. Users who
# actually use yeeHaw wire it in with one line in their own config:
#
#   programs.yh.steeds = config.services.infernix.yeehaw.generatedSteeds;
#
# Or import `infernix.homeModules.yeehaw` alongside yeeHaw's HM module to wire
# that assignment automatically.
#
# Set `services.infernix.yeehaw.enable = true` to populate it; when
# disabled (default), generatedSteeds is {}.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types filterAttrs foldlAttrs;
  cfg = config.services.infernix.yeehaw;
  epCfg = config.services.infernix.endpoints;

  containerUrlOf = ep:
    if ep.containerUrl != null
    then ep.containerUrl
    else if ep.url != null
    then builtins.replaceStrings ["://localhost:" "://127.0.0.1:"] ["://host.docker.internal:" "://host.docker.internal:"] ep.url
    else null;

  safe = s: builtins.replaceStrings ["-" ":" "."] ["_" "_" "_"] s;

  mkOllamaSteed = _epName: ep: model: {
    framework = "goose";
    gooseProvider = "ollama";
    model = model.name;
    # Container-reachable URL; preflight rewrites host.docker.internal → localhost
    # host-side. Mirrors the mkLlamaSwapSteed pattern.
    host = containerUrlOf ep;
    ctxSize = model.ctxSize;
  };

  mkLlamaSwapSteed = epName: ep: model: {
    framework = "goose";
    gooseProvider = "custom";
    engine = "openai";
    host = ep.url;
    baseUrl = containerUrlOf ep;
    model = model.name;
    providerName = "llama_swap_${safe epName}_${safe model.name}";
    blockingGroup = model.blockingGroup;
  };

  mkSteeds = epName: ep:
    foldlAttrs (
      acc: modelKey: model: let
        steedName = "${epName}-${modelKey}";
        steed =
          if ep.type == "ollama"
          then mkOllamaSteed epName ep model
          else mkLlamaSwapSteed epName ep model;
      in
        acc // {${steedName} = filterAttrs (_: v: v != null) steed;}
    ) {}
    ep.models;

  allSteeds = foldlAttrs (acc: epName: ep: acc // mkSteeds epName ep) {} epCfg;
in {
  options.services.infernix.yeehaw = {
    enable = mkEnableOption "auto-generation of yeeHaw steeds from infernix endpoints";

    extraSteeds = mkOption {
      type = types.attrs;
      default = {};
      description = "Additional steeds merged into the generated set.";
    };

    generatedSteeds = mkOption {
      type = types.attrs;
      readOnly = true;
      description = ''
        Steeds auto-generated from local self-hosted
        services.infernix.endpoints.
        Wire this into programs.yh.steeds in your own config:
          programs.yh.steeds = config.services.infernix.yeehaw.generatedSteeds;
      '';
    };
  };

  config.services.infernix.yeehaw.generatedSteeds =
    if cfg.enable
    then allSteeds // cfg.extraSteeds
    else {};
}
