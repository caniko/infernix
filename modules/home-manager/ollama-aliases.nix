# Auto-generates ollama load/unload shell aliases from endpoints.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkIf filterAttrs foldlAttrs nameValuePair;
  epCfg = config.services.infernis.endpoints;

  ollamaEndpoints = filterAttrs (_: ep: ep.type == "ollama") epCfg;

  mkAliases = _epName: ep:
    foldlAttrs (acc: modelKey: model:
      acc
      // {
        "ollama-load-${modelKey}" = "curl -s ${ep.url}/api/generate -d '{\"model\": \"${model.name}\"}'";
        "ollama-unload-${modelKey}" = "curl -s ${ep.url}/api/generate -d '{\"model\": \"${model.name}\", \"keep_alive\": 0}'";
      })
    {}
    ep.models;

  allAliases = foldlAttrs (acc: epName: ep: acc // mkAliases epName ep) {} ollamaEndpoints;
in {
  config = mkIf (ollamaEndpoints != {}) {
    home.shellAliases = allAliases;
  };
}
