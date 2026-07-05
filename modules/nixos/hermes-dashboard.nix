{
  config,
  infernixHermesAgent,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkDefault mkEnableOption mkIf mkOption optionalAttrs types;
  cfg = config.services.infernix.hermes-dashboard;
  system = pkgs.stdenv.hostPlatform.system;

  hermesAgentPackage =
    if builtins.hasAttr "packages" infernixHermesAgent
    && builtins.hasAttr system infernixHermesAgent.packages
    then infernixHermesAgent.packages.${system}.default
    else null;

  oidcEnv =
    optionalAttrs (cfg.publicUrl != null) {
      HERMES_DASHBOARD_PUBLIC_URL = cfg.publicUrl;
    }
    // optionalAttrs (cfg.oidc.issuer != null) {
      HERMES_DASHBOARD_OIDC_ISSUER = cfg.oidc.issuer;
    }
    // optionalAttrs (cfg.oidc.clientId != null) {
      HERMES_DASHBOARD_OIDC_CLIENT_ID = cfg.oidc.clientId;
    }
    // optionalAttrs (cfg.oidc.scopes != []) {
      HERMES_DASHBOARD_OIDC_SCOPES = lib.concatStringsSep " " cfg.oidc.scopes;
    };

  dashboardCommand = pkgs.writeShellScript "hermes-dashboard-start" ''
    set -eu

    ${lib.optionalString (cfg.oidc.clientSecretFile != null) ''
      export HERMES_DASHBOARD_OIDC_CLIENT_SECRET="$(${pkgs.coreutils}/bin/tr -d '\n' < ${cfg.oidc.clientSecretFile})"
    ''}

    exec ${cfg.package}/bin/hermes dashboard \
      --host ${lib.escapeShellArg cfg.host} \
      --port ${toString cfg.port} \
      --no-open \
      ${lib.escapeShellArgs cfg.extraArgs}
  '';
in {
  options.services.infernix.hermes-dashboard = {
    enable = mkEnableOption "Hermes Agent native dashboard via Infernix";

    package = mkOption {
      type = types.nullOr types.package;
      default = hermesAgentPackage;
      defaultText = "inputs.hermes-agent.packages.\${system}.default";
      description = "hermes-agent package. Defaults to the upstream flake package.";
    };

    host = mkOption {
      type = types.str;
      default = "127.0.0.1";
      description = "Address the dashboard binds to.";
    };

    port = mkOption {
      type = types.port;
      default = 9119;
      description = "TCP port the dashboard listens on.";
    };

    publicUrl = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Public dashboard URL used for OIDC callback construction.";
    };

    user = mkOption {
      type = types.str;
      default = "hermes";
      description = "System user the dashboard runs as.";
    };

    group = mkOption {
      type = types.str;
      default = "hermes";
      description = "System group for the dashboard user.";
    };

    createUser = mkOption {
      type = types.bool;
      default = true;
      description = "Create the dashboard user and group automatically.";
    };

    stateDir = mkOption {
      type = types.str;
      default = "/var/lib/hermes";
      description = "State directory whose .hermes subdirectory is HERMES_HOME.";
    };

    workingDirectory = mkOption {
      type = types.str;
      default = "${cfg.stateDir}/workspace";
      defaultText = ''"''${stateDir}/workspace"'';
      description = "Working directory for dashboard-spawned Hermes operations.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
      description = "Additional non-secret environment variables for the dashboard.";
    };

    environmentFiles = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Additional systemd EnvironmentFile paths for KEY=value secrets.";
    };

    extraArgs = mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Extra command-line arguments for hermes dashboard.";
    };

    openFirewall = mkOption {
      type = types.bool;
      default = false;
      description = "Open the firewall for the dashboard port.";
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
        description = "Self-hosted OIDC issuer URL.";
      };

      clientId = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Self-hosted OIDC client ID.";
      };

      scopes = mkOption {
        type = types.listOf types.str;
        default = ["openid" "profile" "email"];
        description = "OIDC scopes requested by the dashboard.";
      };

      clientSecretFile = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "File containing the raw OIDC client secret for confidential clients.";
      };
    };
  };

  config = mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.package != null;
        message = ''
          services.infernix.hermes-dashboard: the upstream hermes-agent package is
          not available for the host system `${system}`. Ensure the hermes-agent
          flake supports this system.
        '';
      }
      {
        assertion = (cfg.oidc.issuer == null) == (cfg.oidc.clientId == null);
        message = "services.infernix.hermes-dashboard: oidc.issuer and oidc.clientId must be set together.";
      }
    ];

    users.groups = mkIf cfg.createUser {
      ${cfg.group} = {};
    };

    users.users = mkIf cfg.createUser {
      ${cfg.user} = {
        isSystemUser = mkDefault true;
        group = mkDefault cfg.group;
        home = mkDefault cfg.stateDir;
        createHome = mkDefault true;
        shell = mkDefault pkgs.bashInteractive;
      };
    };

    systemd.tmpfiles.rules = [
      "d ${cfg.stateDir} 2770 ${cfg.user} ${cfg.group} - -"
      "d ${cfg.stateDir}/.hermes 2770 ${cfg.user} ${cfg.group} - -"
      "d ${cfg.workingDirectory} 2770 ${cfg.user} ${cfg.group} - -"
    ];

    networking.firewall.allowedTCPPorts = mkIf cfg.openFirewall [cfg.port];

    systemd.services.hermes-dashboard = {
      description = "Hermes Agent Dashboard";
      wantedBy = ["multi-user.target"];
      after = ["network-online.target"];
      wants = ["network-online.target"];

      environment =
        {
          HOME = cfg.stateDir;
          HERMES_HOME = "${cfg.stateDir}/.hermes";
          HERMES_MANAGED = "true";
          MESSAGING_CWD = cfg.workingDirectory;
        }
        // oidcEnv
        // cfg.environment;

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
        ProtectHome = false;
        ReadWritePaths = [
          cfg.stateDir
          cfg.workingDirectory
        ];
        PrivateTmp = true;
      };

      path = [
        cfg.package
        pkgs.bash
        pkgs.coreutils
        pkgs.git
      ];
    };
  };
}
