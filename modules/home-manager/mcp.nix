# Compatibility surface. Fleetix owns validation, rendering and reconciliation.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.mcp;
in {
  options.services.infernix.mcp = {
    servers = mkOption {
      type = types.attrsOf (types.submodule {
        options = {
          transport = mkOption {
            type = types.enum ["stdio" "http"];
            default = "stdio";
          };
          command = mkOption {
            type = types.str;
            default = "";
          };
          args = mkOption {
            type = types.listOf types.str;
            default = [];
          };
          url = mkOption {
            type = types.nullOr types.str;
            default = null;
          };
          environment = mkOption {
            type = types.attrsOf types.str;
            default = {};
          };
          key = mkOption {
            type = types.nullOr types.str;
            default = null;
          };
          harnesses = mkOption {
            type = types.nullOr (types.listOf types.str);
            default = null;
          };
        };
      });
      default = {};
      description = "Compatibility registry; new integrations should declare fleetix.mcp.servers.";
    };
    resolvedServers = mkOption {
      type = types.attrs;
      readOnly = true;
    };
  };
  config = {
    services.infernix.mcp.resolvedServers = cfg.servers;
    fleetix.mcp.enable = lib.mkIf (cfg.servers != {}) (lib.mkDefault true);
    fleetix.mcp.servers = lib.mapAttrs' (name: server:
      lib.nameValuePair (
        if server.key == null
        then name
        else server.key
      ) {
        command =
          if server.transport == "stdio"
          then server.command
          else null;
        url =
          if server.transport == "http"
          then server.url
          else null;
        inherit (server) harnesses;
        args =
          if server.transport == "stdio"
          then server.args
          else [];
        env =
          if server.transport == "stdio"
          then server.environment
          else {};
      })
    cfg.servers;
    assertions = [
      {
        assertion = let
          keys = lib.mapAttrsToList (name: server:
            if server.key == null
            then name
            else server.key)
          cfg.servers;
        in
          builtins.length keys == builtins.length (lib.unique keys);
        message = "services.infernix.mcp: duplicate server key overrides; use unique fleetix.mcp.servers names.";
      }
    ];
  };
}
