{ config
, infernixHermesAgent
, lib
, pkgs
, ...
}:
let
  inherit (lib) mkEnableOption mkIf mkMerge mkOption recursiveUpdate types;
  cfg = config.services.infernix.hermes-agent;
  fleetCfg = config.services.infernix.fleet;
  system = pkgs.stdenv.hostPlatform.system;
  renderHermesModelRouting = import ../../lib/hermes-model-routing.nix { inherit lib; };

  hermesAgentPackage =
    if builtins.hasAttr "packages" infernixHermesAgent
      && builtins.hasAttr system infernixHermesAgent.packages
    then infernixHermesAgent.packages.${system}.default
    else null;

  # When useFleetModels is enabled, generate a base_url pointing at the
  # fleet load balancer so hermes uses local GPU-backed models.
  fleetBaseUrl =
    if fleetCfg.loadBalancer.enable
    then "http://${fleetCfg.loadBalancer.host}:${toString fleetCfg.loadBalancer.port}/v1"
    else null;

  legacyFleetSettings = lib.optionalAttrs (cfg.useFleetModels && fleetBaseUrl != null) {
    model.base_url = fleetBaseUrl;
  };

  modelRoutingSettings =
    if cfg.modelRouting.enable
    then
      renderHermesModelRouting
        {
          inherit fleetBaseUrl;
          cloudRouterBaseUrl = config.services.infernix.cloud-router.baseUrl;
          profile = cfg.modelRouting.profile;
        }
    else { };

  baseHermesSettings = recursiveUpdate (recursiveUpdate legacyFleetSettings modelRoutingSettings) cfg.settings;

  instanceSettings = instanceCfg:
    recursiveUpdate
      baseHermesSettings
      (recursiveUpdate
        (if instanceCfg.modelRouting.enable
        then
          renderHermesModelRouting
            {
              inherit fleetBaseUrl;
              cloudRouterBaseUrl = config.services.infernix.cloud-router.baseUrl;
              profile = instanceCfg.modelRouting.profile;
            }
        else { })
        instanceCfg.settings);

  instanceModule = { config, name, ... }: {
    options = {
      enable = mkEnableOption "Hermes Agent instance ${name}";

      package = mkOption {
        type = types.nullOr types.package;
        default = hermesAgentPackage;
        description = "Hermes Agent package for this instance.";
      };

      modelRouting = {
        enable = mkEnableOption "generated Hermes routing for this instance";

        profile = mkOption {
          type = types.attrs;
          default = { };
          description = "Pkl-generated model-routing profile for this instance.";
        };
      };

      user = mkOption {
        type = types.str;
        default = "hermes-${name}";
        description = "System user running this instance.";
      };

      group = mkOption {
        type = types.str;
        default = "hermes-${name}";
        description = "System group running this instance.";
      };

      createUser = mkOption {
        type = types.bool;
        default = true;
        description = "Create the instance user and group.";
      };

      stateDir = mkOption {
        type = types.str;
        default = "/var/lib/hermes-${name}";
        description = "State directory for this instance.";
      };

      workingDirectory = mkOption {
        type = types.str;
        default = "${config.stateDir}/workspace";
        description = "Working directory for this instance.";
      };

      settings = mkOption {
        type = types.attrs;
        default = { };
        description = "Hermes settings for this instance.";
      };

      environment = mkOption {
        type = types.attrsOf types.str;
        default = { };
        description = "Non-secret environment values for this instance.";
      };

      environmentFiles = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "KEY=value files appended to this instance's .env.";
      };

      documents = mkOption {
        type = types.attrsOf (types.either types.str types.path);
        default = { };
        description = "Workspace documents installed for this instance.";
      };

      extraPackages = mkOption {
        type = types.listOf types.package;
        default = [ ];
        description = "Packages exposed to this instance.";
      };

      extraArgs = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra arguments passed to this instance's gateway.";
      };

      allowedToolsets = mkOption {
        type = types.nullOr (types.listOf types.str);
        default = null;
        description = "Hard allowlist of toolsets for this instance.";
      };

      readOnlyState = mkOption {
        type = types.bool;
        default = false;
        description = "Protect this instance's Nix-managed config and environment files.";
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
    };
  };

  scheduledCfg = cfg.scheduledSettings;
  scheduleEnabled = scheduledCfg.enable;
  configYamlMode = if cfg.addToSystemPackages then "0660" else "0640";
  scheduledProfileConfigs =
    lib.mapAttrs
      (name: profile:
        pkgs.writeText "hermes-agent-${name}-config.yaml"
          (builtins.toJSON (recursiveUpdate baseHermesSettings profile.settingsOverlay)))
      scheduledCfg.profiles;
  dailySwitches =
    lib.mapAttrsToList
      (name: switch:
        let
          parseTwoDigits = value:
            (lib.toInt (builtins.substring 0 1 value)) * 10
            + lib.toInt (builtins.substring 1 1 value);
          match = builtins.match "\\*-\\*-\\* ([0-9][0-9]):([0-9][0-9]):([0-9][0-9])" switch.onCalendar;
          hour = parseTwoDigits (builtins.elemAt match 0);
          minute = parseTwoDigits (builtins.elemAt match 1);
          second = parseTwoDigits (builtins.elemAt match 2);
        in
        {
          inherit name;
          inherit (switch) profile onCalendar;
          seconds = hour * 3600 + minute * 60 + second;
        })
      scheduledCfg.switches;
  sortedDailySwitches = lib.sort (a: b: a.seconds < b.seconds) dailySwitches;
  lastDailySwitch =
    if sortedDailySwitches == [ ]
    then null
    else lib.last sortedDailySwitches;
  profileCase = lib.concatStringsSep "\n" (
    lib.mapAttrsToList
      (name: source: ''
        ${lib.escapeShellArg name})
          source=${lib.escapeShellArg source}
          ;;
      '')
      scheduledProfileConfigs
  );
  switchHermesSchedule = pkgs.writeShellScript "hermes-agent-scheduled-settings-switch" ''
    set -eu

    if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
      echo "usage: $0 PROFILE [--restart|--no-restart]" >&2
      exit 64
    fi

    profile="$1"
    restart="''${2:---restart}"

    case "$profile" in
      ${profileCase}
      *)
        echo "hermes-agent schedule: unknown profile '$profile'" >&2
        exit 64
        ;;
    esac

    target=${lib.escapeShellArg "${cfg.stateDir}/.hermes/config.yaml"}
    marker=${lib.escapeShellArg "${cfg.stateDir}/.hermes/.scheduled-settings-profile"}
    tmp="$target.tmp.$$"
    changed=0

    ${pkgs.coreutils}/bin/install -d -o ${lib.escapeShellArg cfg.user} -g ${lib.escapeShellArg cfg.group} -m 2770 ${lib.escapeShellArg "${cfg.stateDir}/.hermes"}
    if ! ${pkgs.diffutils}/bin/cmp -s "$source" "$target"; then
      ${pkgs.coreutils}/bin/install -o ${lib.escapeShellArg cfg.user} -g ${lib.escapeShellArg cfg.group} -m ${configYamlMode} "$source" "$tmp"
      ${pkgs.coreutils}/bin/mv -f "$tmp" "$target"
      changed=1
    fi

    printf '%s\n' "$profile" > "$marker.tmp.$$"
    ${pkgs.coreutils}/bin/chown ${lib.escapeShellArg cfg.user}:${lib.escapeShellArg cfg.group} "$marker.tmp.$$"
    ${pkgs.coreutils}/bin/chmod 0644 "$marker.tmp.$$"
    ${pkgs.coreutils}/bin/mv -f "$marker.tmp.$$" "$marker"

    echo "hermes-agent schedule: active_profile=$profile changed=$changed config=$target"

    if [ "$restart" = "--restart" ] && [ "$changed" -eq 1 ] && [ ${lib.escapeShellArg (lib.boolToString scheduledCfg.restartService)} = "true" ]; then
      ${pkgs.systemd}/bin/systemctl try-restart hermes-agent.service
    fi
  '';
  bootstrapHermesSchedule = pkgs.writeShellScript "hermes-agent-scheduled-settings-bootstrap" ''
    set -eu

    now_h="$(TZ=${lib.escapeShellArg scheduledCfg.timeZone} ${pkgs.coreutils}/bin/date +%H)"
    now_m="$(TZ=${lib.escapeShellArg scheduledCfg.timeZone} ${pkgs.coreutils}/bin/date +%M)"
    now_s="$(TZ=${lib.escapeShellArg scheduledCfg.timeZone} ${pkgs.coreutils}/bin/date +%S)"
    now_seconds=$((10#$now_h * 3600 + 10#$now_m * 60 + 10#$now_s))
    profile=${lib.escapeShellArg (if lastDailySwitch == null then "" else lastDailySwitch.profile)}

    ${lib.concatMapStringsSep "\n" (switch: ''
      if [ "$now_seconds" -ge ${toString switch.seconds} ]; then
        profile=${lib.escapeShellArg switch.profile}
      fi
    '') sortedDailySwitches}

    if [ -z "$profile" ]; then
      echo "hermes-agent schedule: no bootstrap profile could be selected" >&2
      exit 1
    fi

    exec ${switchHermesSchedule} "$profile" --no-restart
  '';
in
{
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
        URL and exposes fleet-declared models. Prefer modelRouting for
        mixed cloud/local Hermes configurations.
      '';
    };

    modelRouting = {
      enable = mkEnableOption "generated Hermes provider, alias, fallback, and auxiliary model routing";

      profile = mkOption {
        type = types.attrs;
        default = { };
        description = ''
          Pkl-generated Hermes model-routing profile. The renderer turns this
          into settings.model, custom_providers, model_aliases, fallback_model,
          and auxiliary.
        '';
      };
    };

    scheduledSettings = {
      enable = mkEnableOption "scheduled Hermes settings overlays";

      timeZone = mkOption {
        type = types.str;
        default = "UTC";
        description = "IANA timezone used by scheduled Hermes settings timers and bootstrap profile selection.";
      };

      restartService = mkOption {
        type = types.bool;
        default = true;
        description = "Restart hermes-agent.service after a scheduled profile switch changes config.yaml.";
      };

      profiles = mkOption {
        type = types.attrsOf (types.submodule {
          options.settingsOverlay = mkOption {
            type = types.attrs;
            default = { };
            description = "Hermes settings overlay recursively merged over the normal generated settings for this profile.";
          };
        });
        default = { };
        description = "Named scheduled Hermes settings profiles.";
      };

      switches = mkOption {
        type = types.attrsOf (types.submodule {
          options = {
            profile = mkOption {
              type = types.str;
              description = "Scheduled settings profile activated by this switch.";
            };

            onCalendar = mkOption {
              type = types.str;
              example = "*-*-* 09:00:00";
              description = ''
                Daily systemd OnCalendar expression without timezone. The
                configured scheduledSettings.timeZone is appended for the timer
                and used for bootstrap selection.
              '';
            };
          };
        });
        default = { };
        description = "Daily scheduled switches for Hermes settings profiles.";
      };
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
      default = { };
      description = "Declarative hermes config rendered as config.yaml. Deep-merged across module definitions.";
    };

    environmentFiles = mkOption {
      type = types.listOf types.str;
      default = [ ];
      description = "Paths to env files with secrets, merged into HERMES_HOME/.env.";
    };

    environment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Non-secret env vars. Do NOT put secrets here.";
    };

    documents = mkOption {
      type = types.attrsOf (types.either types.str types.path);
      default = { };
      description = "Workspace files (SOUL.md, USER.md, etc.) installed into workingDirectory.";
    };

    mcpServers = mkOption {
      type = types.attrs;
      default = { };
      description = "MCP server definitions merged into settings.mcp_servers.";
    };

    extraPackages = mkOption {
      type = types.listOf types.package;
      default = [ ];
      description = "Extra packages available to the agent.";
    };

    extraPlugins = mkOption {
      type = types.listOf types.package;
      default = [ ];
      description = "Directory-based plugin packages symlinked into the hermes plugins dir.";
    };

    extraPythonPackages = mkOption {
      type = types.listOf types.package;
      default = [ ];
      description = "Python packages added to PYTHONPATH for entry-point plugin discovery.";
    };

    extraDependencyGroups = mkOption {
      type = types.listOf types.str;
      default = [ ];
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
      default = [ ];
      description = "Extra command-line arguments for hermes gateway.";
    };

    allowedToolsets = mkOption {
      type = types.nullOr (types.listOf types.str);
      default = null;
      description = "Hard allowlist of toolsets for the singleton service.";
    };

    readOnlyState = mkOption {
      type = types.bool;
      default = false;
      description = "Protect the singleton's Nix-managed config and environment files.";
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
        type = types.enum [ "docker" "podman" ];
        default = "docker";
        description = "Container runtime.";
      };

      extraVolumes = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra volume mounts (host:container:mode).";
      };

      extraOptions = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Extra arguments passed to docker/podman create.";
      };

      image = mkOption {
        type = types.str;
        default = "ubuntu:24.04";
        description = "OCI container image.";
      };

      hostUsers = mkOption {
        type = types.listOf types.str;
        default = [ ];
        description = "Interactive users who get a ~/.hermes symlink to the service stateDir.";
      };
    };

    instances = mkOption {
      type = types.attrsOf (types.submodule instanceModule);
      default = { };
      description = "Independent native Hermes Agent instances.";
    };
  };

  config = mkMerge [
    (mkIf cfg.enable (mkMerge [
      {
        assertions = [
          {
            assertion = cfg.package != null;
            message = ''
              services.infernix.hermes-agent: the upstream hermes-agent package is
              not available for the host system `${system}`. Ensure the
              hermes-agent flake supports this system.
            '';
          }
        ]
        ++ lib.optional scheduleEnabled {
          assertion = scheduledCfg.profiles != { };
          message = "services.infernix.hermes-agent.scheduledSettings: at least one profile is required when enabled.";
        }
        ++ lib.optional scheduleEnabled {
          assertion = scheduledCfg.switches != { };
          message = "services.infernix.hermes-agent.scheduledSettings: at least one switch is required when enabled.";
        }
        ++ lib.optionals scheduleEnabled (
          lib.mapAttrsToList
            (name: switch: {
              assertion = builtins.hasAttr switch.profile scheduledCfg.profiles;
              message = "services.infernix.hermes-agent.scheduledSettings.switches.${name}: profile '${switch.profile}' is not defined.";
            })
            scheduledCfg.switches
        )
        ++ lib.optionals scheduleEnabled (
          lib.mapAttrsToList
            (name: switch: {
              assertion = builtins.match "\\*-\\*-\\* ([0-9][0-9]):([0-9][0-9]):([0-9][0-9])" switch.onCalendar != null;
              message = "services.infernix.hermes-agent.scheduledSettings.switches.${name}.onCalendar must use daily '*-*-* HH:MM:SS' form.";
            })
            scheduledCfg.switches
        );

        services.hermes-agent = {
          enable = true;
          package = cfg.package;

          inherit (cfg) user group stateDir addToSystemPackages
            environmentFiles environment documents extraPackages
            extraPlugins extraPythonPackages extraDependencyGroups
            configFile authFile authFileForceOverwrite extraArgs restart restartSec
            allowedToolsets readOnlyState;

          settings = baseHermesSettings;

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
      }

      (mkIf scheduleEnabled {
        systemd.services =
          {
            hermes-agent-scheduled-settings-bootstrap = {
              description = "Select scheduled Hermes settings profile";
              before = [ "hermes-agent.service" ];
              wantedBy = [ "multi-user.target" ];
              serviceConfig = {
                Type = "oneshot";
                ExecStart = "${bootstrapHermesSchedule}";
              };
            };

            hermes-agent = {
              after = [ "hermes-agent-scheduled-settings-bootstrap.service" ];
              requires = [ "hermes-agent-scheduled-settings-bootstrap.service" ];
            };
          }
          // lib.mapAttrs'
            (name: switch:
              lib.nameValuePair "hermes-agent-scheduled-settings-${name}" {
                description = "Switch Hermes settings profile to ${switch.profile}";
                serviceConfig = {
                  Type = "oneshot";
                  ExecStart = "${switchHermesSchedule} ${lib.escapeShellArg switch.profile} --restart";
                };
              })
            scheduledCfg.switches;

        systemd.timers = lib.mapAttrs'
          (name: switch:
            lib.nameValuePair "hermes-agent-scheduled-settings-${name}" {
              description = "Activate Hermes settings profile ${switch.profile}";
              wantedBy = [ "timers.target" ];
              timerConfig = {
                OnCalendar = "${switch.onCalendar} ${scheduledCfg.timeZone}";
                Persistent = true;
                Unit = "hermes-agent-scheduled-settings-${name}.service";
              };
            })
          scheduledCfg.switches;
      })
    ]))

    {
      assertions = lib.concatLists (lib.mapAttrsToList
        (name: instanceCfg:
          lib.optional (instanceCfg.enable && instanceCfg.package == null) {
            assertion = false;
            message = "services.infernix.hermes-agent.instances.${name}: no Hermes package is available for this host system `${system}`.";
          })
        cfg.instances);

      services.hermes-agent.instances = lib.mkMerge (lib.mapAttrsToList
        (name: instanceCfg:
          lib.optionalAttrs instanceCfg.enable {
            "${name}" = {
              enable = true;
              package = instanceCfg.package;
              inherit (instanceCfg)
                user
                group
                createUser
                stateDir
                workingDirectory
                restart
                restartSec
                ;
              environmentFiles = cfg.environmentFiles ++ instanceCfg.environmentFiles;
              environment = cfg.environment // instanceCfg.environment;
              documents = cfg.documents // instanceCfg.documents;
              extraPackages = cfg.extraPackages ++ instanceCfg.extraPackages;
              extraArgs = cfg.extraArgs ++ instanceCfg.extraArgs;
              allowedToolsets =
                if instanceCfg.allowedToolsets != null
                then instanceCfg.allowedToolsets
                else cfg.allowedToolsets;
              readOnlyState = cfg.readOnlyState || instanceCfg.readOnlyState;
              settings = instanceSettings instanceCfg;
            };
          })
        cfg.instances);
    }
  ];
}
