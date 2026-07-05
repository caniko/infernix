{ lib }:

let
  inherit (lib) attrValues filterAttrs mapAttrs optionalAttrs removeAttrs replaceStrings;

  stripPkl = attrs:
    if builtins.isAttrs attrs
    then removeAttrs attrs [ "__pkl_class" ]
    else attrs;

  modelsOf = catalog: catalog.models or { };

  modelsForHost = catalog: host:
    filterAttrs (_: model: (model.host or null) == host) (modelsOf catalog);

  renderArg = modelsDir: arg:
    replaceStrings [ "{modelsDir}" ] [ (toString modelsDir) ] arg;

  renderLlamaSwapModel = modelsDir: name: model:
    let
      cleaned = stripPkl model;
    in
    {
      repo = cleaned.repo;
      file = cleaned.file;
      ctxSize = cleaned.ctxSize;
      ttl = cleaned.ttl or 300;
      aliases = cleaned.aliases or [ ];
      extraArgs = map (renderArg modelsDir) (cleaned.extraArgs or [ ]);
      extraFiles = map stripPkl (cleaned.extraFiles or [ ]);
    }
    // optionalAttrs (cleaned ? draft && cleaned.draft != null) {
      draft = stripPkl cleaned.draft;
    };

  hasLlamaSwapPayload = model:
    model ? repo && model ? file && model ? ctxSize;

  renderFleetModel = name: model: {
    name = model.name or name;
    aliases = model.fleetAliases or (model.aliases or [ ]);
    capabilities = model.capabilities or [ ];
  };

  resolveEndpointModel = catalog: spec:
    let
      modelName = spec.model;
      model = (modelsOf catalog).${modelName} or { };
    in
    {
      name = spec.name or (model.name or modelName);
      ctxSize = spec.ctxSize or model.ctxSize;
    };
in
{
  mkLlamaSwapModels = { catalog, host, modelsDir }:
    mapAttrs (renderLlamaSwapModel modelsDir)
      (filterAttrs (_: hasLlamaSwapPayload) (modelsForHost catalog host));

  mkFleetModels = { catalog, host }:
    mapAttrs renderFleetModel
      (filterAttrs (_: model: model ? capabilities) (modelsForHost catalog host));

  mkHmEndpointModels = { catalog, endpoint }:
    mapAttrs (_: resolveEndpointModel catalog)
      (catalog.homeManager.endpoints.${endpoint}.models or { });

  mkProbeSpecs = { catalog, host }:
    catalog.probes.${host} or { };

  modelNamesForHost = { catalog, host }:
    map (model: model.name or null) (attrValues (modelsForHost catalog host));
}
