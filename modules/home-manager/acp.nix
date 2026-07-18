# Shared Agent Client Protocol provider registry.
#
# Consumers use one typed provider declaration instead of each maintaining its
# own binary, arguments, environment, and safety policy.  The default Codex
# provider is the official agentclientprotocol/codex-acp adapter, wrapped to
# reuse the Nix-provided Codex executable.
{ config
, lib
, pkgs
, infernixCodexAcp ? null
, ...
}:
let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.acp;
  defaultCommand =
    if infernixCodexAcp != null
    then "${infernixCodexAcp}/bin/codex-acp"
    else "codex-acp";

  providerSubmodule = types.submodule ({ name, ... }: {
    options = {
      package = mkOption {
        type = types.nullOr types.package;
        default = if name == "codex" then infernixCodexAcp else null;
        description = "Optional package installed for this ACP provider.";
      };
      command = mkOption {
        type = types.str;
        default = if name == "codex" then defaultCommand else name;
        description = "ACP agent command or absolute path.";
      };
      args = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Arguments passed to the ACP agent.";
      };
      environment = mkOption {
        type = types.attrsOf types.str;
        default = if name == "codex" then {
          CODEX_HOME = "${config.home.homeDirectory}/.codex";
          CODEX_PATH = lib.getExe pkgs.codex;
        } else { };
        description = "Environment inherited by the ACP agent.";
      };
      configOptions = mkOption {
        type = types.attrsOf (types.oneOf [ types.str types.bool ]);
        default = { };
        description = "Provider-wide ACP session option defaults.";
      };
      capabilities = mkOption {
        type = types.attrsOf types.bool;
        default = if name == "codex" then {
          image = true;
          sessionConfig = true;
          text = true;
        } else {
          text = true;
        };
        description = "ACP capabilities available to consumers of this provider.";
      };
    };
  });
in {
  options.services.infernix.acp = {
    enable = mkEnableOption "shared ACP providers" // { default = true; };

    providers = mkOption {
      type = types.attrsOf providerSubmodule;
      default = { codex = { }; };
      description = ''
        Typed ACP provider registry shared by Graphify, visual-rubric, Zed,
        and other consumers. Provider settings are data; consumers do not
        duplicate adapter-specific command-line flags.
      '';
    };

    resolvedProviders = mkOption {
      type = types.attrsOf providerSubmodule;
      readOnly = true;
      description = "Fully resolved ACP provider commands, packages, environment, and capabilities.";
    };
  };

  config = {
    services.infernix.acp.resolvedProviders = cfg.providers;
    assertions = lib.optionals cfg.enable [
      {
        assertion = lib.all (provider: provider.command != "") (lib.attrValues cfg.providers);
        message = "services.infernix.acp.providers entries must resolve to a command.";
      }
    ];
    home.packages = lib.mkIf cfg.enable (
      lib.unique (lib.filter (package: package != null) (map (provider: provider.package) (lib.attrValues cfg.resolvedProviders)))
    );
  };
}
