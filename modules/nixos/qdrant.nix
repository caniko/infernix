{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types;
  cfg = config.services.infernix.qdrant;
in {
  options.services.infernix.qdrant = {
    enable = mkEnableOption "Qdrant vector database";

    host = mkOption {
      type = types.str;
      default = "0.0.0.0";
      description = "Address to bind Qdrant to.";
    };

    httpPort = mkOption {
      type = types.port;
      default = 6333;
      description = "HTTP API port.";
    };

    grpcPort = mkOption {
      type = types.port;
      default = 6334;
      description = "gRPC API port.";
    };

    storagePath = mkOption {
      type = types.path;
      default = "/var/lib/qdrant/storage";
      description = "Path to store Qdrant collections.";
    };

    snapshotsPath = mkOption {
      type = types.path;
      default = "/var/lib/qdrant/snapshots";
      description = "Path to store Qdrant snapshots.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = true;
      description = "Whether to open firewall ports for Qdrant.";
    };
  };

  config = mkIf cfg.enable {
    services.qdrant = {
      enable = true;
      settings = {
        service = {
          host = cfg.host;
          http_port = cfg.httpPort;
          grpc_port = cfg.grpcPort;
        };
        storage = {
          storage_path = toString cfg.storagePath;
          snapshots_path = toString cfg.snapshotsPath;
        };
      };
    };

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [cfg.httpPort cfg.grpcPort];
  };
}
