{
  pkgs,
  homeManager,
}: let
  inherit (pkgs) lib;
  sample = homeManager.lib.homeManagerConfiguration {
    inherit pkgs;
    modules = [
      ../modules/home-manager/skill-telemetry.nix
      {
        home = {
          username = "tester";
          homeDirectory = "/home/tester";
          stateVersion = "24.11";
        };
        services.infernix.skillTelemetry = {
          enable = true;
          skillnetPackage = pkgs.hello;
          harnesses =
            lib.mapAttrs (_: nativeEvent: {
              enable = true;
              inherit nativeEvent;
            })
            nativeEvents
            // {disabled.enable = false;};
        };
      }
    ];
  };
  nativeEvents = {
    claude = "PostToolUse";
    codex = "AfterToolUse";
    gemini = "AfterTool";
    opencode = "tool.execute.after";
  };
  exe = builtins.unsafeDiscardStringContext (lib.getExe pkgs.hello);
  telemetry = config: config.services.infernix.skillTelemetry;
  base = telemetry sample.config;
  withTriggers = options:
    (sample.extendModules {
      modules = [{services.infernix.skillTelemetry.cardTriggers = {enable = true;} // options;}];
    }).config;
  enabled = withTriggers {};
  packsDir = "/var/lib/card packs";
  packed = withTriggers {packsDir = packsDir;};
  activation = sample.config.home.activation.infernixSkillTelemetryClaude.data;
in
  assert (lib.hasInfix ''if [ -x "${exe}" ]'' activation);
  assert (lib.hasInfix ''"${exe}" hook install'' activation);
  assert builtins.attrNames base.generatedAdapters == builtins.attrNames nativeEvents;
  assert lib.all (harness: let
    adapter = base.generatedAdapters.${harness};
  in
    adapter.ingestCommand
    == "${exe} hook ingest --harness ${harness} --event ${nativeEvents.${harness}}"
    && lib.hasPrefix "${exe} usage record --harness ${harness} " adapter.recordCommand)
  (builtins.attrNames nativeEvents);
  assert !base.cardTriggers.enable && base.cardTriggers.packsDir == null && base.cardTriggers.events == ["post-edit"];
  assert base.generatedCardTriggerAdapters == {};
  assert builtins.attrNames (telemetry enabled).generatedCardTriggerAdapters == builtins.attrNames nativeEvents;
  assert lib.all (harness: (telemetry enabled).generatedCardTriggerAdapters.${harness}.triggerCommand == "${exe} trigger --event post-edit --harness ${harness} --json") (builtins.attrNames nativeEvents);
  assert (telemetry packed).generatedCardTriggerAdapters.claude.triggerCommand == "${exe} trigger --event post-edit --harness claude --json --packs-dir '/var/lib/card packs'";
  assert (telemetry (withTriggers {events = [];})).generatedCardTriggerAdapters == {};
  assert (telemetry enabled).generatedAdapters == base.generatedAdapters;
  assert enabled.home.activation == sample.config.home.activation;
  assert builtins.attrNames enabled.home.file == builtins.attrNames sample.config.home.file; true
