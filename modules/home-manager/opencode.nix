{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.infernix.modelProviders;
  opencodeWrapper = pkgs.writeShellApplication {
    name = "opencode";
    text = ''
      ${import ../../lib/codex-credentials.nix {
        inherit lib;
        package = cfg.codexProviderPackage;
        keyFile = cfg.codexProviderKeyFile;
      }}
      exec ${lib.getExe config.programs.opencode.package} "$@"
    '';
  };
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
      inherit (value) id;
      tool_call = value.tool_call or true;
    }
    // lib.optionalAttrs (value ? reasoning) {inherit (value) reasoning;}
    // lib.optionalAttrs (value ? limit) {inherit (value) limit;};

  toProvider = _name: value: {
    npm = "@ai-sdk/openai-compatible";
    inherit (value) name;
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
  services.infernix.modelProviders.codexProviderEnable = lib.mkDefault (cfg.providers ? codex);
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
  home.packages = lib.optional (cfg.providers ? codex) (lib.hiPrio opencodeWrapper);
}
