# Shared skill-usage integration. Infernix owns harness wiring; Skillnet owns
# persistence and reporting. Harnesses must provide a native activation event.
{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.skillTelemetry;
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
      recordCommand = lib.optionalString (cfg.skillnetPackage != null) "${lib.getExe cfg.skillnetPackage} usage record --harness ${lib.escapeShellArg name} --adapter-version ${lib.escapeShellArg value.adapterVersion} --skill <canonical-skill> --session <session-hash> --event-id <source-event-id>";
    })
    enabledHarnesses;
in {
  options.services.infernix.skillTelemetry = {
    enable = mkEnableOption "native skill usage recording through Skillnet";

    skillnetPackage = mkOption {
      type = types.nullOr types.package;
      default = null;
      description = "Skillnet package providing the usage recorder.";
    };

    harnesses = mkOption {
      type = types.attrsOf harnessSubmodule;
      default = {};
      description = "Harness adapters. Every enabled adapter must declare a native event.";
    };

    generatedAdapters = mkOption {
      type = types.attrs;
      readOnly = true;
      description = "Read-only adapter metadata and normalized recorder commands.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = lib.optional (cfg.skillnetPackage != null) cfg.skillnetPackage;
    home.activation.infernixSkillTelemetryClaude = lib.mkIf claudeEnabled (lib.hm.dag.entryAfter ["writeBoundary"] ''
      if command -v skillnet >/dev/null 2>&1; then
        skillnet hook install --settings "$HOME/.claude/settings.json" --events PostToolUse --matchers Skill
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
  };
}
