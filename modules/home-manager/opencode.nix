{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.infernix.modelProviders;
  # CCR selectors use provider,model; OpenCode uses provider/model. Replace
  # only the selector separator, preserving slashes inside model IDs.
  modelRef = selector: let
    parts = lib.splitString "," selector;
  in
    if builtins.length parts == 2
    then "${builtins.head parts}/${builtins.elemAt parts 1}"
    else selector;

  toModel = _id: value:
    {
      name = value.name or value.id;
      id = value.id;
      tool_call = value.tool_call or true;
    }
    // lib.optionalAttrs (value ? reasoning) {reasoning = value.reasoning;}
    // lib.optionalAttrs (value ? limit) {limit = value.limit;};

  toProvider = _name: value: {
    npm = "@ai-sdk/openai-compatible";
    name = value.name;
    options = {
      baseURL = value.baseUrl;
      apiKey =
        if value ? apiKeyEnv
        then "{env:${value.apiKeyEnv}}"
        else value.apiKey;
      timeout = 600000;
      chunkTimeout = 30000;
    };
    models = lib.mapAttrs toModel value.models;
  };
in {
  programs.opencode = {
    enable = lib.mkDefault true;
    package = lib.mkDefault pkgs.opencode;
    settings = {
      model = modelRef cfg.routes.default;
      small_model = modelRef cfg.routes.background;
      agent = {
        build.model = modelRef cfg.routes.background;
        plan.model = modelRef cfg.routes.think;
        general = {
          model = modelRef cfg.routes.default;
          variant = "medium";
        };
      };
      provider = lib.mapAttrs toProvider cfg.providers;
    };
  };
}
