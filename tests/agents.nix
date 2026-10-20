{
  pkgs,
  homeManager,
}: let
  inherit
    ((homeManager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        ../modules/home-manager/agents.nix
        # A second path import must deduplicate (homeModules.agents next to default).
        ../modules/home-manager/agents.nix
        {
          home = {
            username = "tester";
            homeDirectory = "/home/tester";
            stateVersion = "24.11";
          };
          services.infernix.agents = {
            scout = {
              description = "Read-only: lookup";
              steps = 6;
              prompt = "Stay bounded.";
              permission = {
                edit = "deny";
                task = "deny";
                bash = "ask";
              };
              opencode.model = "openai/sol";
              claude = {
                model = "haiku";
                effort = "medium";
              };
            };
            title.description = "OpenCode-only";
          };
        }
      ];
    }))
    config
    ;
  oc = config.programs.opencode.settings.agent;
in
  assert oc.scout.model == "openai/sol" && oc.scout.steps == 6 && oc.scout.mode == "subagent";
  assert oc.scout.permission.bash == "ask";
  assert oc.title
  == {
    description = "OpenCode-only";
    mode = "subagent";
    permission = {};
  };
  assert !(config.home.file ? ".claude/agents/title.md");
  assert config.home.file.".claude/agents/scout.md".text
  == ''
    ---
    name: "scout"
    description: "Read-only: lookup"
    model: "haiku"
    effort: "medium"
    maxTurns: 6
    disallowedTools: "Edit, Write, NotebookEdit, Agent"
    ---
    Stay bounded.
  ''; true
