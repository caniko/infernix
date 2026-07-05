{ lib }:

{ profile
, fleetBaseUrl
, cloudRouterBaseUrl
,
}:
let
  providerBaseUrl = provider:
    if provider.urlSource == "fleetLoadBalancer"
    then
      if fleetBaseUrl != null
      then fleetBaseUrl
      else throw "services.infernix.hermes-agent.modelRouting: provider uses fleetLoadBalancer but services.infernix.fleet.loadBalancer.enable is false"
    else if provider.urlSource == "cloudRouter"
    then cloudRouterBaseUrl
    else if provider.urlSource == "literal"
    then provider.baseUrl
    else throw "services.infernix.hermes-agent.modelRouting: unsupported provider urlSource '${provider.urlSource}'";

  renderProvider = name: provider:
    {
      inherit name;
      base_url = providerBaseUrl provider;
      api_key = provider.apiKey or "no-key-required";
      models = provider.models or { };
    }
    // lib.optionalAttrs (provider ? defaultModel && provider.defaultModel != null) {
      model = provider.defaultModel;
    };
in
{
  model = profile.model or { };
  custom_providers = lib.mapAttrsToList renderProvider (profile.providers or { });
  model_aliases = profile.modelAliases or { };
  fallback_model = profile.fallbackModel or [ ];
  auxiliary = profile.auxiliary or { };
}
