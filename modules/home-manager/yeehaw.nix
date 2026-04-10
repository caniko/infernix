# Auto-generates yeeHaw steeds from infernis endpoints.
#
# Opt-in: set `services.infernis.yeehaw.enable = true` on users who have
# yeeHaw's HM module imported (programs.yh option available).
#
# When disabled, this module does not reference `programs.yh.*` at all,
# so it is safe to include in shared modules for users who don't use yeeHaw.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types filterAttrs foldlAttrs;
  cfg = config.services.infernis.yeehaw;
  epCfg = config.services.infernis.endpoints;

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
    model = model.name;
    host = ep.url;
    ctxSize = model.ctxSize;
  };

  mkLlamaSwapSteed = epName: ep: model: let
    cUrl = containerUrlOf ep;
    safe = s: builtins.replaceStrings ["-" ":" "."] ["_" "_" "_"] s;
  in {
    provider = "goose";
    backend = "custom";
    engine = "openai";
    host = ep.url;
    baseUrl = cUrl;
    model = model.name;
    providerName = "llama_swap_${safe epName}_${safe model.name}";
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
        acc // {${steedName} = filterAttrs (_: v: v != null) steed;}
    ) {}
    ep.models;

  allSteeds = foldlAttrs (acc: epName: ep: acc // mkSteeds epName ep) {} epCfg;
in {
  options.services.infernis.yeehaw = {
    enable = mkEnableOption "auto-generate yeeHaw steeds from infernis endpoints";

    extraSteeds = mkOption {
      type = types.attrs;
      default = {};
      description = "Additional steeds merged into the generated set.";
    };
  };

  # Plain `if-then-else` (not mkIf) so that when disabled, the attribute
  # path `programs.yh.steeds` is truly absent — avoiding type-check errors
  # for users who don't have yeeHaw's HM module imported.
  config =
    if cfg.enable
    then {
      programs.yh.steeds = allSteeds // cfg.extraSteeds;
    }
    else {};
}
