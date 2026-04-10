# Auto-generates yeeHaw steeds from infernis endpoints.
# Requires yeeHaw's HM module to be imported separately by the consumer.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkIf filterAttrs foldlAttrs mapAttrs';
  epCfg = config.services.infernis.endpoints;
  hasEndpoints = epCfg != {};

  # Derive containerUrl from url if not explicitly set
  containerUrlOf = ep:
    if ep.containerUrl != null
    then ep.containerUrl
    else if ep.url != null
    then builtins.replaceStrings ["://localhost:" "://127.0.0.1:"] ["://host.docker.internal:" "://host.docker.internal:"] ep.url
    else null;

  mkOllamaSteed = _epName: ep: model: {
    provider = "goose";
    backend = "ollama";
    inherit (model) name;
    model = model.name;
    host = ep.url;
    ctxSize = model.ctxSize;
  };

  mkLlamaSwapSteed = _epName: ep: model: let
    cUrl = containerUrlOf ep;
  in {
    provider = "goose";
    backend = "custom";
    engine = "openai";
    host = ep.url;
    baseUrl = cUrl;
    model = model.name;
    providerName = "llama_swap_${builtins.replaceStrings ["-"] ["_"] _epName}_${builtins.replaceStrings ["-"] ["_"] model.name}";
    blockingGroup = model.blockingGroup;
  };

  mkClaudeSteed = _epName: _ep: model: {
    provider = "claude";
    model = model.name;
  };

  mkSteeds = epName: ep:
    foldlAttrs (
      acc: modelKey: model: let
        steedName = "${epName}-${modelKey}";
        steed =
          if ep.type == "ollama"
          then mkOllamaSteed epName ep model
          else if ep.type == "llama-swap"
          then mkLlamaSwapSteed epName ep model
          else mkClaudeSteed epName ep model;
      in
        acc // {${steedName} = lib.filterAttrs (_: v: v != null) steed;}
    ) {}
    ep.models;

  allSteeds = foldlAttrs (acc: epName: ep: acc // mkSteeds epName ep) {} epCfg;
in {
  config = mkIf (hasEndpoints && (config.programs ? yh) && config.programs.yh.enable) {
    programs.yh.steeds = allSteeds;
  };
}
