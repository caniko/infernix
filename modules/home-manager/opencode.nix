{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.infernix.modelProviders;

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
      model = cfg.routes.default;
      small_model = cfg.routes.background;
      agent = {
        build.model = cfg.routes.background;
        plan.model = cfg.routes.think;
        general = {
          model = cfg.routes.default;
          variant = "medium";
        };
      };
      provider = lib.mapAttrs toProvider cfg.providers;
    };
  };
}
