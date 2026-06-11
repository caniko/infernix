{
  config,
  infernixSelf,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.loadBalancer;

  modelSubmodule = types.submodule {
    options = {
      name = mkOption {
        type = types.str;
        description = "Backend model name sent to the upstream OpenAI-compatible server.";
      };

      aliases = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional public model names accepted by the load balancer.";
      };

      capabilities = mkOption {
        type = types.listOf (types.enum ["chat" "embeddings" "rerank"]);
        default = [];
        description = "OpenAI API capabilities this backend model can serve.";
      };
    };
  };

  backendSubmodule = types.submodule {
    options = {
      baseUrl = mkOption {
        type = types.str;
        example = "http://192.168.178.29:8013";
        description = "Base URL for the backend OpenAI-compatible model server.";
      };

      healthUrl = mkOption {
        type = types.str;
        example = "http://192.168.178.29:8020/healthz";
        description = "Health URL for the host-local infernix node service.";
      };

      priority = mkOption {
        type = types.ints.unsigned;
        default = 100;
        description = "Lower values are preferred before in-flight load tie-breaking.";
      };

      weight = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Tie-breaker weight; larger values win after priority and in-flight count.";
      };

      maxInFlight = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Maximum concurrent proxied requests for this backend.";
      };

      models = mkOption {
        type = types.attrsOf modelSubmodule;
        default = {};
        description = "Models this backend is allowed to serve.";
      };
    };
  };

  configFile = (pkgs.formats.toml {}).generate "infernix-lb.toml" {
    listen = {
      inherit (cfg) host port;
    };
    request_timeout_secs = cfg.requestTimeoutSecs;
    backends =
      lib.mapAttrsToList (id: backend: {
        inherit id;
        base_url = backend.baseUrl;
        health_url = backend.healthUrl;
        inherit (backend) priority weight;
        max_in_flight = backend.maxInFlight;
        models =
          lib.mapAttrsToList (modelId: model: {
            id = modelId;
            inherit (model) name aliases capabilities;
          })
          backend.models;
      })
      cfg.backends;
  };
in {
  options.services.infernix.loadBalancer = {
    enable = mkEnableOption "Infernix OpenAI-compatible model load balancer";

    package = mkOption {
      type = types.package;
      default = infernixSelf.packages.${pkgs.stdenv.hostPlatform.system}.infernix-lb;
      defaultText = lib.literalExpression "inputs.infernix.packages.<system>.infernix-lb";
      description = "infernix-lb package to run.";
    };

    host = mkOption {
      type = types.str;
      default = "192.168.178.31";
      description = "Address to bind the load balancer to.";
    };

    port = mkOption {
      type = types.port;
      default = 8014;
      description = "Port to bind the load balancer to.";
    };

    requestTimeoutSecs = mkOption {
      type = types.ints.positive;
      default = 300;
      description = "Backend request timeout in seconds.";
    };

    backends = mkOption {
      type = types.attrsOf backendSubmodule;
      default = {};
      description = "Backend GPU hosts and their model capabilities.";
    };

    openFirewallInterfaces = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["end0"];
      description = "Network interfaces where the load-balancer TCP port is opened.";
    };
  };

  config = mkIf cfg.enable {
    systemd.services.infernix-lb = {
      description = "Infernix OpenAI-compatible GPU load balancer";
      after = ["network-online.target"];
      wants = ["network-online.target"];
      wantedBy = ["multi-user.target"];
      serviceConfig = {
        ExecStart = "${cfg.package}/bin/infernix-lb --config ${configFile}";
        DynamicUser = true;
        Restart = "on-failure";
        RestartSec = 5;
        NoNewPrivileges = true;
        ProtectHome = true;
        ProtectSystem = "strict";
        PrivateTmp = true;
      };
    };

    networking.firewall.interfaces = lib.genAttrs cfg.openFirewallInterfaces (_: {
      allowedTCPPorts = [cfg.port];
    });
  };
}
