{
  config,
  infernixHermesAgent,
  lib,
  pkgs,
  ...
}: let
      inherit (lib) mkEnableOption mkIf mkOption recursiveUpdate types;
  cfg = config.services.infernix.hermes-agent;
  fleetCfg = config.services.infernix.fleet;
  system = pkgs.stdenv.hostPlatform.system;

  hermesAgentPackage =
    if builtins.hasAttr "packages" infernixHermesAgent
    && builtins.hasAttr system infernixHermesAgent.packages
    then infernixHermesAgent.packages.${system}.default
    else null;

  # When useFleetModels is enabled, generate a base_url pointing at the
  # fleet load balancer so hermes uses local GPU-backed models.
  fleetBaseUrl =
    if cfg.useFleetModels && fleetCfg.loadBalancer.enable
    then "http://${fleetCfg.loadBalancer.host}:${toString fleetCfg.loadBalancer.port}/v1"
    else null;
in {
  options.services.infernix.hermes-agent = {
    enable = mkEnableOption "Hermes Agent gateway service via Infernix";

    package = mkOption {
      type = types.nullOr types.package;
      default = hermesAgentPackage;
      defaultText = "inputs.hermes-agent.packages.\${system}.default";
      description = "hermes-agent package. Defaults to the upstream flake package.";
    };

    useFleetModels = mkOption {
      type = types.bool;
      default = false;
      description = ''
        Auto-configure hermes model endpoints from the Infernix fleet
        load balancer when it is enabled. Sets model.base_url to the LB
        URL and exposes fleet-declared models.
      '';
    };

    # Pass-through options that map 1:1 to the upstream service.hermes-agent.
    user = mkOption {
      type = types.str;
      default = "hermes";
      description = "System user the gateway runs as.";
    };

    group = mkOption {
      type = types.str;
      default = "hermes";
      description = "System group for the gateway user.";
    };

    stateDir = mkOption {
      type = types.str;
      default = "/var/lib/hermes";
      description = "State directory (HERMES_HOME parent).";
    };

    addToSystemPackages = mkOption {
      type = types.bool;
      default = false;
      description = "Add hermes CLI to system PATH and set HERMES_HOME system-wide.";
    };

    settings = mkOption {
      type = types.attrs;
      default = {};
      description = "Declarative hermes config rendered as config.yaml. Deep-merged across module definitions.";
    };

    environmentFiles = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Paths to env files with secrets, merged into HERMES_HOME/.env.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Non-secret env vars. Do NOT put secrets here.";
    };

    documents = mkOption {
      type = types.attrsOf (types.either types.str types.path);
      default = {};
      description = "Workspace files (SOUL.md, USER.md, etc.) installed into workingDirectory.";
    };

    mcpServers = mkOption {
      type = types.attrs;
      default = {};
      description = "MCP server definitions merged into settings.mcp_servers.";
    };

    extraPackages = mkOption {
      type = types.listOf types.package;
      default = [];
      description = "Extra packages available to the agent.";
    };

    extraPlugins = mkOption {
      type = types.listOf types.package;
      default = [];
      description = "Directory-based plugin packages symlinked into the hermes plugins dir.";
    };

    extraPythonPackages = mkOption {
      type = types.listOf types.package;
      default = [];
      description = "Python packages added to PYTHONPATH for entry-point plugin discovery.";
    };

    extraDependencyGroups = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "pyproject.toml optional extras included in the sealed venv.";
    };

    configFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to an existing config.yaml. Overrides settings entirely.";
    };

    authFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = "Path to an auth.json seed file (OAuth credentials).";
    };

    authFileForceOverwrite = mkOption {
      type = types.bool;
      default = false;
      description = "Always overwrite auth.json from authFile on activation.";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments for hermes gateway.";
    };

    restart = mkOption {
      type = types.str;
      default = "always";
      description = "systemd Restart= policy.";
    };

    restartSec = mkOption {
      type = types.int;
      default = 5;
      description = "systemd RestartSec= value.";
    };

    container = {
      enable = mkEnableOption "OCI container mode for hermes-agent";

      backend = mkOption {
        type = types.enum ["docker" "podman"];
        default = "docker";
        description = "Container runtime.";
      };

      extraVolumes = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Extra volume mounts (host:container:mode).";
      };

      extraOptions = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Extra arguments passed to docker/podman create.";
      };

      image = mkOption {
        type = types.str;
        default = "ubuntu:24.04";
        description = "OCI container image.";
      };

      hostUsers = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Interactive users who get a ~/.hermes symlink to the service stateDir.";
      };
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.package != null;
        message = ''
          services.infernix.hermes-agent: the upstream hermes-agent package is
          not available for the host system `${system}`. Ensure the
          hermes-agent flake supports this system.
        '';
      }
    ];

    services.hermes-agent = {
      enable = true;
      package = cfg.package;

      inherit (cfg) user group stateDir addToSystemPackages
        environmentFiles environment documents extraPackages
        extraPlugins extraPythonPackages extraDependencyGroups
        configFile authFile authFileForceOverwrite extraArgs restart restartSec;

      settings = recursiveUpdate cfg.settings (
        lib.optionalAttrs (fleetBaseUrl != null) {
          model.base_url = fleetBaseUrl;
        }
      );

      mcpServers = cfg.mcpServers;

      container = {
        enable = cfg.container.enable;
        backend = cfg.container.backend;
        extraVolumes = cfg.container.extraVolumes;
        extraOptions = cfg.container.extraOptions;
        image = cfg.container.image;
        hostUsers = cfg.container.hostUsers;
      };
    };
  };
}
