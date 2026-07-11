# Expose an OpenAI-compatible Infernix endpoint for downstream Graphify
# semantic extraction. Infernix owns endpoint/model routing; Graphify only
# receives the standard OPENAI_BASE_URL and OPENAI_MODEL environment values.
{config, lib, ...}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.graphify;
  endpoints = config.services.infernix.endpoints;
  endpoint = endpoints.${cfg.endpoint} or null;
  model =
    if endpoint == null || cfg.model == null
    then null
    else endpoint.models.${cfg.model} or null;
  baseUrl =
    if endpoint == null || endpoint.url == null
    then null
    else if lib.hasSuffix "/v1" endpoint.url
    then endpoint.url
    else "${endpoint.url}/v1";
  generatedSettings =
    if !cfg.enable || endpoint == null || model == null || baseUrl == null
    then {}
    else {
      OPENAI_BASE_URL = baseUrl;
      OPENAI_MODEL = model.name;
      # The OpenAI SDK requires a non-empty key even when the local endpoint
      # does not authenticate requests. Infernix never sends this placeholder
      # outside the configured local endpoint.
      OPENAI_API_KEY = "sk-infernix-local";
    };
in {
  options.services.infernix.graphify = {
    enable = mkEnableOption "Graphify semantic extraction through Infernix";

    endpoint = mkOption {
      type = types.str;
      default = "infernix-lb";
      description = "Endpoint name from services.infernix.endpoints.";
    };

    model = mkOption {
      type = types.str;
      default = "dsv4";
      description = "Model key from the selected Infernix endpoint.";
    };

    generatedSettings = mkOption {
      type = types.attrsOf types.str;
      readOnly = true;
      description = ''
        OpenAI-compatible environment values for Graphify semantic extraction.
        Consumers should pass these to Graphify rather than duplicating endpoint
        or model routing policy.
      '';
    };
  };

  config = {
    services.infernix.graphify.generatedSettings = generatedSettings;

    assertions = lib.optionals cfg.enable [
      {
        assertion = endpoints ? ${cfg.endpoint};
        message = "services.infernix.graphify.endpoint refers to '${cfg.endpoint}', which is not declared in services.infernix.endpoints.";
      }
      {
        assertion = endpoint == null || endpoint.type == "llama-swap";
        message = "services.infernix.graphify.endpoint '${cfg.endpoint}' must be a llama-swap endpoint (OpenAI-compatible).";
      }
      {
        assertion = endpoint == null || endpoint.models ? ${cfg.model};
        message = "services.infernix.graphify.model '${cfg.model}' is not defined on endpoint '${cfg.endpoint}'.";
      }
    ];
  };
}
