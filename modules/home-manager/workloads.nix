{ config
, lib
, ...
}:
let
  inherit (lib) mkOption types;
  endpoints = config.services.infernix.endpoints;

  routeSubmodule = types.submodule {
    options = {
      endpoint = mkOption {
        type = types.str;
        description = "Endpoint name from services.infernix.endpoints.";
      };

      model = mkOption {
        type = types.str;
        description = "Model key on the selected endpoint.";
      };
    };
  };

  profileSubmodule = types.submodule ({ name, ... }: {
    options = {
      routing = {
        endpoint = mkOption {
          type = types.str;
          description = "Primary endpoint name.";
        };

        model = mkOption {
          type = types.str;
          description = "Primary model key.";
        };

        capability = mkOption {
          type = types.enum [ "chat" "embeddings" "rerank" ];
          description = "Capability required from every selected route.";
        };

        locality = mkOption {
          type = types.enum [ "local-only" "network-allowed" ];
          default = "local-only";
          description = "Network locality permitted by this workload.";
        };

        dataResidency = mkOption {
          type = types.enum [ "local-only" "eu" "ch" "us" "unrestricted" ];
          default = "local-only";
          description = "Data-residency requirement for this workload.";
        };

        endpointTypes = mkOption {
          type = types.listOf (types.enum [ "ollama" "llama-swap" ]);
          default = [ "llama-swap" ];
          description = "Endpoint protocols accepted by this workload.";
        };

        healthAware = mkOption {
          type = types.bool;
          default = true;
          description = "Require endpoint health before selecting a route.";
        };

        timeoutSecs = mkOption {
          type = types.ints.positive;
          default = 300;
          description = "Bounded request timeout for the workload route.";
        };

        retry = {
          maxAttempts = mkOption {
            type = types.ints.positive;
            default = 1;
            description = "Maximum route attempts, including the primary.";
          };

          backoffSecs = mkOption {
            type = types.ints.unsigned;
            default = 0;
            description = "Delay between retry/fallback attempts.";
          };
        };

        fallback = mkOption {
          type = types.nullOr routeSubmodule;
          default = null;
          description = "Optional deterministic fallback route.";
        };

        credentialRef = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = ''
            Opaque credential reference. The reference is never included in
            resolved output; credentials must be injected by the consumer.
          '';
        };
      };

      execution = {
        adapter = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Logical adapter name; executable paths stay outside this profile.";
        };

        queues = mkOption {
          type = types.listOf types.str;
          default = [ ];
          description = "Queues the logical adapter may execute.";
        };
      };

      lease = {
        enabled = mkOption {
          type = types.bool;
          default = false;
          description = "Enable durable lease/fencing for this workload.";
        };

        concurrency = mkOption {
          type = types.ints.positive;
          default = 1;
          description = "Maximum concurrent leases for this profile.";
        };

        durationSecs = mkOption {
          type = types.ints.positive;
          default = 900;
          description = "Lease duration in seconds.";
        };

        heartbeatSecs = mkOption {
          type = types.ints.positive;
          default = 30;
          description = "Lease heartbeat interval in seconds.";
        };

        maxAttempts = mkOption {
          type = types.ints.positive;
          default = 3;
          description = "Maximum execution attempts for one desired job.";
        };
      };
    };
  });

  route = endpointName: modelKey:
    let
      endpoint = endpoints.${endpointName} or null;
      model = if endpoint == null then null else endpoint.models.${modelKey} or null;
      baseUrl =
        if endpoint == null || endpoint.url == null
        then null
        else if lib.hasSuffix "/v1" endpoint.url
        then endpoint.url
        else "${endpoint.url}/v1";
    in
    {
      endpoint = endpointName;
      inherit baseUrl;
      modelKey = modelKey;
      model = if model == null then null else model.name;
      healthUrl = if endpoint == null then null else endpoint.healthUrl;
      endpointType = if endpoint == null then null else endpoint.type;
      capabilities = if model == null then [ ] else model.capabilities;
      locality = if endpoint == null then null else endpoint.locality;
      dataResidency = if endpoint == null then null else endpoint.dataResidency;
      apiKeyRequired = if endpoint == null then false else endpoint.apiKeyRequired;
    };

  resolvedProfile = name: profile:
    let
      primary = route profile.routing.endpoint profile.routing.model;
      fallback =
        if profile.routing.fallback == null
        then null
        else route profile.routing.fallback.endpoint profile.routing.fallback.model;
      resolvedRoute = selected: {
        endpoint = selected.endpoint;
        baseUrl = selected.baseUrl;
        model = selected.model;
        capability = profile.routing.capability;
        healthUrl = selected.healthUrl;
        locality = selected.locality;
        dataResidency = selected.dataResidency;
        apiKeyRequired = selected.apiKeyRequired;
      };
    in
    {
      schemaVersion = 1;
      workload = name;
      routing = {
        primary = resolvedRoute primary;
        fallback = if fallback == null then null else resolvedRoute fallback;
        capability = profile.routing.capability;
        locality = profile.routing.locality;
        dataResidency = profile.routing.dataResidency;
        healthAware = profile.routing.healthAware;
        timeoutSecs = profile.routing.timeoutSecs;
        retry = profile.routing.retry;
        credentialRequired =
          profile.routing.credentialRef != null
          || primary.apiKeyRequired
          || (fallback != null && fallback.apiKeyRequired);
      };
      execution = profile.execution;
      lease = profile.lease;
    };

  routeAssertions = profileName: routeName: selected: profile:
    let
      endpoint = endpoints.${selected.endpoint} or null;
      model = if endpoint == null then null else endpoint.models.${selected.modelKey} or null;
    in
    [
      {
        assertion = endpoint != null;
        message = "services.infernix.workloads.${profileName}.${routeName}.endpoint refers to '${selected.endpoint}', which is not declared in services.infernix.endpoints.";
      }
      {
        assertion = endpoint == null || builtins.elem endpoint.type profile.routing.endpointTypes;
        message = "services.infernix.workloads.${profileName}.${routeName}.endpoint '${selected.endpoint}' uses an unsupported endpoint type.";
      }
      {
        assertion = model != null;
        message = "services.infernix.workloads.${profileName}.${routeName}.model '${selected.modelKey}' is not defined on endpoint '${selected.endpoint}'.";
      }
      {
        assertion = selected.baseUrl != null;
        message = "services.infernix.workloads.${profileName}.${routeName}.endpoint '${selected.endpoint}' must provide a URL.";
      }
      {
        assertion = model == null || builtins.elem profile.routing.capability model.capabilities;
        message = "services.infernix.workloads.${profileName}.${routeName} requires capability '${profile.routing.capability}', which the model does not advertise.";
      }
      {
        assertion = endpoint == null || profile.routing.locality != "local-only" || endpoint.locality == "local-only";
        message = "services.infernix.workloads.${profileName}.${routeName} violates the local-only locality policy.";
      }
      {
        assertion = endpoint == null || profile.routing.dataResidency == "unrestricted" || endpoint.dataResidency == profile.routing.dataResidency;
        message = "services.infernix.workloads.${profileName}.${routeName} violates the data-residency policy.";
      }
      {
        assertion = !profile.routing.healthAware || selected.healthUrl != null;
        message = "services.infernix.workloads.${profileName}.${routeName} requires a healthUrl for health-aware routing.";
      }
    ];

  profileAssertions = lib.concatLists (lib.mapAttrsToList
    (name: profile:
      let
        primary = route profile.routing.endpoint profile.routing.model;
        fallback =
          if profile.routing.fallback == null
          then null
          else route profile.routing.fallback.endpoint profile.routing.fallback.model;
      in
      routeAssertions name "primary" primary profile
      ++ lib.optional (fallback != null) {
        assertion = profile.routing.retry.maxAttempts > 1;
        message = "services.infernix.workloads.${name}.routing.retry.maxAttempts must be greater than one when fallback is configured.";
      }
      ++ lib.optionals (fallback != null) (routeAssertions name "fallback" fallback profile)
      ++ [
        {
          assertion = profile.execution.adapter != null || profile.execution.queues == [ ];
          message = "services.infernix.workloads.${name}.execution.queues requires an adapter.";
        }
        {
          assertion = !profile.lease.enabled || profile.execution.adapter != null;
          message = "services.infernix.workloads.${name}.lease.enabled requires an execution adapter.";
        }
        {
          assertion = !profile.lease.enabled || profile.execution.queues != [ ];
          message = "services.infernix.workloads.${name}.lease.enabled requires at least one execution queue.";
        }
        {
          assertion = !profile.lease.enabled || profile.lease.heartbeatSecs < profile.lease.durationSecs;
          message = "services.infernix.workloads.${name}.lease.heartbeatSecs must be less than durationSecs.";
        }
      ]
    )
    config.services.infernix.workloads);
in
{
  options.services.infernix.workloads = mkOption {
    type = types.attrsOf profileSubmodule;
    default = { };
    description = ''
      Generic typed workload profiles. Profiles resolve endpoint/model
      capabilities and health-aware fallback routes into a non-secret read-only
      projection; source, prompt, credential, and executable command contents
      are not part of the projection.
    '';
  };

  options.services.infernix.resolvedWorkloads = mkOption {
    type = types.attrsOf types.attrs;
    readOnly = true;
    description = "Resolved non-secret workload routing, execution, and lease profiles.";
  };

  config = {
    assertions = profileAssertions;
    services.infernix.resolvedWorkloads =
      lib.mapAttrs resolvedProfile config.services.infernix.workloads;
  };
}
