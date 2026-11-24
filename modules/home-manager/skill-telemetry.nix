# Shared skill-usage integration. Infernix owns harness wiring; Skillnet owns
# persistence and reporting. Harnesses must provide a native activation event.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.skillTelemetry;
  skillnet = lib.getExe cfg.skillnetPackage;
  harnessSubmodule = types.submodule ({name, ...}: {
    options = {
      enable = mkEnableOption "Skillnet usage recording for ${name}";
      nativeEvent = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Native harness event that represents a skill activation.";
      };
      adapterVersion = mkOption {
        type = types.str;
        default = "1";
        description = "Version of the harness adapter emitting usage events.";
      };
    };
  });
  enabledHarnesses = lib.filterAttrs (_: value: value.enable) cfg.harnesses;
  claudeEnabled = (enabledHarnesses.claude or null) != null;
  generated =
    lib.mapAttrs (name: value: {
      harness = name;
      event = value.nativeEvent;
      adapterVersion = value.adapterVersion;
      recordCommand = lib.optionalString (cfg.skillnetPackage != null) "${skillnet} usage record --harness ${lib.escapeShellArg name} --adapter-version ${lib.escapeShellArg value.adapterVersion} --skill <canonical-skill> --session <session-hash> --event-id <source-event-id>";
      ingestCommand = lib.optionalString (cfg.skillnetPackage != null && value.nativeEvent != null) "${skillnet} hook ingest --harness ${lib.escapeShellArg name} --event ${lib.escapeShellArg value.nativeEvent}";
    })
    enabledHarnesses;
in {
  options.services.infernix.skillTelemetry = {
    enable = mkEnableOption "native skill usage recording through Skillnet";

    skillnetPackage = mkOption {
      type = types.nullOr types.package;
      default = null;
      description = "Skillnet package providing usage recording, hook ingestion, and optional card triggers.";
    };

    harnesses = mkOption {
      type = types.attrsOf harnessSubmodule;
      default = {};
      description = "Harness adapters. Every enabled adapter must declare a native event.";
    };

    generatedAdapters = mkOption {
      type = types.attrs;
      readOnly = true;
      description = "Read-only adapter metadata and commands for usage recording and native hook ingestion.";
    };

    cardTriggers = mkOption {
      type = types.submodule {
        options = {
          enable = mkEnableOption "card trigger commands for enabled Skillnet harness adapters";
          packsDir = mkOption {
            type = types.nullOr types.path;
            default = null;
            description = "Optional absolute path to the card packs directory. Null uses Skillnet's own lookup.";
          };
          events = mkOption {
            type = types.listOf (types.enum ["post-edit"]);
            default = ["post-edit"];
            description = "Card trigger events to expose. Only post-edit is wired for harness adapters.";
          };
        };
      };
      default = {};
      description = "Opt-in card triggers. Harness integrations decide how to register the generated commands.";
    };

    generatedCardTriggerAdapters = mkOption {
      type = types.attrs;
      readOnly = true;
      description = "Read-only card trigger commands by harness. Does not modify harness settings.";
    };
  };

  config = lib.mkMerge [
    {
      services.infernix.skillTelemetry.generatedCardTriggerAdapters =
        lib.optionalAttrs (cfg.enable && cfg.cardTriggers.enable && cfg.skillnetPackage != null && builtins.elem "post-edit" cfg.cardTriggers.events)
        (lib.mapAttrs (harness: _: {
            triggerCommand =
              "${skillnet} trigger --event post-edit --harness ${lib.escapeShellArg harness} --json"
              + lib.optionalString (cfg.cardTriggers.packsDir != null) " --packs-dir ${lib.escapeShellArg "${cfg.cardTriggers.packsDir}"}";
          })
          enabledHarnesses);
    }
    (lib.mkIf cfg.enable {
      home.packages = lib.optional (cfg.skillnetPackage != null) cfg.skillnetPackage;
      home.activation.infernixSkillTelemetryClaude = lib.mkIf (claudeEnabled && cfg.skillnetPackage != null) (lib.hm.dag.entryAfter ["writeBoundary"] ''
        if [ -x "${skillnet}" ]; then
          "${skillnet}" hook install --settings "$HOME/.claude/settings.json" --events PostToolUse --matchers Skill
        fi
      '');
      services.infernix.skillTelemetry.generatedAdapters = generated;
      assertions =
        [
          {
            assertion = cfg.skillnetPackage != null;
            message = "services.infernix.skillTelemetry.skillnetPackage must be set when skill telemetry is enabled.";
          }
        ]
        ++ lib.mapAttrsToList (name: value: {
          assertion = value.nativeEvent != null;
          message = "services.infernix.skillTelemetry.harnesses.${name} requires a native activation event; heuristic transcript parsing is not supported.";
        })
        enabledHarnesses;
    })
  ];
}
