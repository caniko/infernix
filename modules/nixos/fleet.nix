{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkMerge mkOption types;
  cfg = config.services.infernix.fleet;

  modelSubmodule = types.submodule {
    options = {
      name = mkOption {
        type = types.str;
        description = "Backend model name sent to the upstream model server.";
      };

      aliases = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Additional public model names accepted by the load balancer.";
      };

      capabilities = mkOption {
        type = types.listOf (types.enum ["chat" "embeddings" "rerank"]);
        default = [];
        description = "Model capabilities exposed through the Infernix load balancer.";
      };
    };
  };

  nodeSubmodule = types.submodule {
    options = {
      lanIp = mkOption {
        type = types.str;
        description = "LAN IP address for the GPU backend host.";
      };

      modelPort = mkOption {
        type = types.port;
        default = 8013;
        description = "OpenAI-compatible backend model server port.";
      };

      nodePort = mkOption {
        type = types.port;
        default = 8020;
        description = "Infernix node health/control endpoint port.";
      };

      priority = mkOption {
        type = types.ints.unsigned;
        default = 100;
        description = "Lower values are preferred by the load balancer.";
      };

      weight = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Load-balancer tie-breaker weight.";
      };

      maxInFlight = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Maximum concurrent proxied requests for this backend.";
      };

      units = mkOption {
        type = types.listOf types.str;
        default = ["llama-swap.service"];
        description = "Systemd units controlled by infernix-nodectl on this node.";
      };

      models = mkOption {
        type = types.attrsOf modelSubmodule;
        default = {};
        description = "Models this backend is allowed to serve.";
      };
    };
  };

  lbBackends =
    lib.mapAttrs (_name: node: {
      baseUrl = "http://${node.lanIp}:${toString node.modelPort}";
      healthUrl = "http://${node.lanIp}:${toString node.nodePort}/healthz";
      inherit (node) priority weight maxInFlight models;
    })
    cfg.nodes;

  localNode =
    if cfg.localNodeName == null
    then null
    else cfg.nodes.${cfg.localNodeName};
in {
  options.services.infernix.fleet = {
    nodes = mkOption {
      type = types.attrsOf nodeSubmodule;
      default = {};
      description = "Declarative GPU backend fleet inventory.";
    };

    localNodeName = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Name of the fleet node represented by this host.";
    };

    loadBalancer = {
      enable = mkEnableOption "generation of services.infernix.loadBalancer.backends from the fleet";

      host = mkOption {
        type = types.str;
        default = "192.168.178.31";
        description = "Address where the generated load balancer binds.";
      };

      port = mkOption {
        type = types.port;
        default = 8014;
        description = "Port where the generated load balancer binds.";
      };

      openFirewallInterfaces = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Interfaces where the generated load-balancer port is opened.";
      };
    };
  };

  config = mkMerge [
    {
      assertions = [
        {
          assertion = cfg.localNodeName == null || builtins.hasAttr cfg.localNodeName cfg.nodes;
          message = "services.infernix.fleet.localNodeName must refer to a key in services.infernix.fleet.nodes.";
        }
      ];
    }

    (mkIf cfg.loadBalancer.enable {
      services.infernix.loadBalancer = {
        enable = true;
        inherit (cfg.loadBalancer) host port openFirewallInterfaces;
        backends = lbBackends;
      };
    })

    (mkIf (cfg.localNodeName != null) {
      services.infernix.node = {
        enable = true;
        host = localNode.lanIp;
        port = localNode.nodePort;
        units = localNode.units;
      };
    })
  ];
}
