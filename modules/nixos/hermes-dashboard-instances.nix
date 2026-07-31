{ config
, infernixHermesAgent
, lib
, pkgs
, ...
}:
let
  inherit (lib) mkIf mkMerge mkOption optionalAttrs types;
  cfg = config.services.infernix.hermes-dashboard;
  system = pkgs.stdenv.hostPlatform.system;
  defaultPackage =
    if builtins.hasAttr "packages" infernixHermesAgent
      && builtins.hasAttr system infernixHermesAgent.packages
    then infernixHermesAgent.packages.${system}.default
    else null;

  instanceModule = { name, ... }: {
    options = {
      enable = lib.mkEnableOption "Hermes dashboard instance ${name}";

      package = mkOption {
        type = types.nullOr types.package;
        default = defaultPackage;
        description = "Hermes package for this dashboard.";
      };

      host = mkOption {
        type = types.str;
        default = "127.0.0.1";
        description = "Address this dashboard binds to.";
      };

      port = mkOption {
        type = types.port;
        description = "TCP port for this dashboard.";
      };

      publicUrl = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Public URL used for authentication callbacks.";
      };

      user = mkOption {
        type = types.str;
        default = "hermes-${name}";
        description = "System user running this dashboard.";
      };

      group = mkOption {
        type = types.str;
        default = "hermes-${name}";
        description = "System group running this dashboard.";
      };

      createUser = mkOption {
        type = types.bool;
        default = true;
        description = "Create the dashboard user and group.";
      };

      stateDir = mkOption {
        type = types.str;
        default = "/var/lib/hermes-${name}";
        description = "State directory whose .hermes subdirectory is managed.";
      };

      workingDirectory = mkOption {
        type = types.str;
        default = "/var/lib/hermes-${name}/workspace";
        description = "Working directory for dashboard chat sessions.";
      };

      environment = mkOption {
        type = types.attrsOf types.str;
        default = { };
        description = "Additional non-secret dashboard environment.";
      };

      environmentFiles = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Secret environment files for this dashboard.";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra arguments passed to Hermes dashboard.";
      };

      openFirewall = mkOption {
        type = types.bool;
        default = false;
        description = "Open the firewall for this dashboard port.";
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

      oidc = {
        issuer = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "OIDC issuer URL.";
        };

        clientId = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "OIDC client ID.";
        };

        scopes = mkOption {
          type = types.listOf types.str;
          default = [ "openid" "profile" "email" ];
          description = "OIDC scopes requested by this dashboard.";
        };

        clientSecretFile = mkOption {
          type = types.nullOr types.path;
          default = null;
          description = "File containing the raw OIDC client secret.";
        };
      };
    };
  };

  mkInstance = name: cfg:
    let
      unitName = "hermes-dashboard-${name}";
      oidcEnv =
        optionalAttrs (cfg.publicUrl != null)
          {
            HERMES_DASHBOARD_PUBLIC_URL = cfg.publicUrl;
          }
        // optionalAttrs (cfg.oidc.issuer != null) {
          HERMES_DASHBOARD_OIDC_ISSUER = cfg.oidc.issuer;
        }
        // optionalAttrs (cfg.oidc.clientId != null) {
          HERMES_DASHBOARD_OIDC_CLIENT_ID = cfg.oidc.clientId;
        }
        // optionalAttrs (cfg.oidc.scopes != [ ]) {
          HERMES_DASHBOARD_OIDC_SCOPES = lib.concatStringsSep " " cfg.oidc.scopes;
        };
      dashboardCommand = pkgs.writeShellScript "${unitName}-start" ''
        set -eu
        ${lib.optionalString (cfg.oidc.clientSecretFile != null) ''
          export HERMES_DASHBOARD_OIDC_CLIENT_SECRET="$(${pkgs.coreutils}/bin/tr -d '\n' < ${lib.escapeShellArg (toString cfg.oidc.clientSecretFile)})"
        ''}
        exec ${cfg.package}/bin/hermes dashboard \
          --host ${lib.escapeShellArg cfg.host} \
          --port ${toString cfg.port} \
          --no-open \
          ${lib.escapeShellArgs cfg.extraArgs}
      '';
    in
    if cfg.enable then {
      assertions = [
        {
          assertion = builtins.match "^[a-z0-9][a-z0-9-]*$" name != null;
          message = "services.infernix.hermes-dashboard.instances.${name}: instance names must be lowercase letters, digits, and hyphens.";
        }
        {
          assertion = cfg.package != null;
          message = "services.infernix.hermes-dashboard.instances.${name}: no Hermes package is available for this host system.";
        }
        {
          assertion = (cfg.oidc.issuer == null) == (cfg.oidc.clientId == null);
          message = "services.infernix.hermes-dashboard.instances.${name}: oidc.issuer and oidc.clientId must be set together.";
        }
      ];

      users.groups = mkIf cfg.createUser {
        ${cfg.group} = { };
      };
      users.users = mkIf cfg.createUser {
        ${cfg.user} = {
          isSystemUser = true;
          group = cfg.group;
          home = cfg.stateDir;
          createHome = true;
          shell = pkgs.bashInteractive;
        };
      };

      systemd.tmpfiles.rules = [
        "d ${cfg.stateDir} 2770 ${cfg.user} ${cfg.group} - -"
        "d ${cfg.stateDir}/.hermes 2770 ${cfg.user} ${cfg.group} - -"
        "d ${cfg.workingDirectory} 2770 ${cfg.user} ${cfg.group} - -"
      ];

      networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [ cfg.port ];

      systemd.services.${unitName} = {
        description = "Hermes Agent ${name} dashboard";
        wantedBy = [ "multi-user.target" ];
        after = [ "network-online.target" ];
        wants = [ "network-online.target" ];
        environment = {
          HOME = cfg.stateDir;
          HERMES_HOME = "${cfg.stateDir}/.hermes";
          HERMES_MANAGED = "true";
          MESSAGING_CWD = cfg.workingDirectory;
        } // oidcEnv // cfg.environment;
        serviceConfig = {
          User = cfg.user;
          Group = cfg.group;
          WorkingDirectory = cfg.workingDirectory;
          EnvironmentFile = cfg.environmentFiles;
          ExecStart = dashboardCommand;
          Restart = cfg.restart;
          RestartSec = cfg.restartSec;
          UMask = "0007";
          NoNewPrivileges = true;
          ProtectSystem = "strict";
          ProtectHome = "read-only";
          ReadWritePaths = [ cfg.stateDir cfg.workingDirectory ];
          PrivateTmp = true;
        };
        path = [
          cfg.package
          pkgs.bash
          pkgs.coreutils
          pkgs.git
        ];
      };
    } else { };
  instanceConfigs = lib.mapAttrsToList mkInstance cfg.instances;
in
{
  options.services.infernix.hermes-dashboard.instances = mkOption {
    type = types.attrsOf (types.submodule instanceModule);
    default = { };
    description = "Independent Hermes dashboard instances.";
  };

  config = {
    assertions = lib.concatMap (instance: instance.assertions or [ ]) instanceConfigs;
    networking = lib.mkMerge (map (instance: instance.networking or { }) instanceConfigs);
    systemd = lib.mkMerge (map (instance: instance.systemd or { }) instanceConfigs);
    users = lib.mkMerge (map (instance: instance.users or { }) instanceConfigs);
  };
}
