# Legacy activation policy translated into Fleetix's standalone MCP catalogue.
{
  config,
  lib,
  infernixFleetixLib,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.harnesses;
  catalogue = infernixFleetixLib.mcp.catalogue;
  hasServers = config.services.infernix.mcp.servers != {};
in {
  options.services.infernix.harnesses = mkOption {
    default = {};
    type = types.attrsOf (types.submodule {
      options = {
        mode = mkOption {
          type = types.enum ["auto" "force" "off"];
          default = "auto";
        };
        probe.commands = mkOption {
          type = types.listOf types.str;
          default = [];
        };
        unsupported = mkOption {
          type = types.nullOr types.str;
          default = null;
        };
        dialect = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Explicit Fleetix dialect for a custom MCP harness.";
        };
        adapters.mcp = mkOption {
          type = types.nullOr (types.submodule {
            options = {
              format = mkOption {type = types.enum ["json" "toml"];};
              configPath = mkOption {type = types.str;};
              root = mkOption {
                type = types.listOf types.str;
                default = ["mcpServers"];
              };
              serverKey = mkOption {
                type = types.nullOr types.str;
                default = null;
              };
            };
          });
          default = null;
          description = "Deprecated destination override; server rendering is always owned by Fleetix.";
        };
      };
    });
  };
  options.services.infernix.harnessRegistry = {
    statusFile = mkOption {
      type = types.str;
      default = config.fleetix.mcp.stateFile;
      description = "Deprecated; use fleetix.mcp.stateFile for the managed-entry ledger.";
    };
    plan = mkOption {
      type = types.attrs;
      readOnly = true;
    };
  };
  config = {
    services.infernix.harnesses = lib.genAttrs ["claude" "codex" "gemini" "hermes" "kiro" "opencode" "copilot"] (name: {
      probe.commands = lib.mkDefault catalogue.${name}.commands;
    });
    services.infernix.harnessRegistry.plan = config.fleetix.mcp.manifest;
    fleetix.mcp.harnesses = lib.mkIf hasServers (lib.mapAttrs (name: harness:
      {
        enable = lib.mkDefault (hasServers && harness.mode != "off" && harness.unsupported == null);
        dialect = lib.mkDefault (
          if harness.dialect == null
          then name
          else harness.dialect
        );
        delivery = lib.mkDefault (
          if name == "hermes"
          then "export"
          else "merge"
        );
        autoDetect = lib.mkDefault (harness.mode == "auto" && name != "hermes");
        commands = lib.mkDefault harness.probe.commands;
      }
      // lib.optionalAttrs (harness.adapters.mcp != null) {
        configPath = lib.mkDefault (lib.replaceStrings ["~/"] ["${config.home.homeDirectory}/"] harness.adapters.mcp.configPath);
        format = lib.mkDefault harness.adapters.mcp.format;
        root = lib.mkDefault harness.adapters.mcp.root;
      }) (lib.filterAttrs (name: h: h.unsupported == null && (builtins.hasAttr name catalogue || h.dialect != null)) cfg));
    assertions = lib.optionals hasServers (lib.concatLists (lib.mapAttrsToList (name: harness: [
        {
          assertion = harness.adapters.mcp == null || harness.unsupported != null || builtins.hasAttr name catalogue || harness.dialect != null;
          message = "Infernix harness '${name}' needs an explicit Fleetix dialect; configure fleetix.mcp.harnesses directly.";
        }
        {
          assertion = harness.adapters.mcp == null || harness.adapters.mcp.serverKey == null;
          message = "Infernix harness '${name}': adapter-wide serverKey is ambiguous; use services.infernix.mcp.servers.<name>.key.";
        }
      ])
      cfg));
  };
}
