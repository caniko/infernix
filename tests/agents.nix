{
  pkgs,
  homeManager,
}: let
  eval = agents:
    (homeManager.lib.homeManagerConfiguration {
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
          services.infernix.agents = agents;
        }
      ];
    }).config;
  config = eval {
    scout = {
      description = "Read-only: lookup";
      steps = 6;
      prompt = "Stay bounded.";
      permission = {
        edit = "deny";
        task = "deny";
        bash = "deny";
        # Unmapped permissions are fine unless denied.
        doom_loop = "ask";
      };
      opencode.model = "openai/sol";
      claude = {
        model = "haiku";
        effort = "medium";
      };
    };
    title.description = "OpenCode-only";
  };
  # A denial Claude Code cannot express must fail evaluation, not widen access.
  unmapped = eval {
    unmapped = {
      description = "Denies a permission Claude cannot express";
      permission.doom_loop = "deny";
      claude.model = "haiku";
    };
  };
  oc = config.programs.opencode.settings.agent;
in
  assert oc.scout.model == "openai/sol" && oc.scout.steps == 6 && oc.scout.mode == "subagent";
  assert oc.scout.permission.doom_loop == "ask";
  assert oc.title
  == {
    description = "OpenCode-only";
    mode = "subagent";
    permission = {};
  };
  assert !(config.home.file ? ".claude/agents/title.md");
  assert !(builtins.tryEval unmapped.home.file.".claude/agents/unmapped.md".text).success;
  assert config.home.file.".claude/agents/scout.md".text
  == ''
    ---
    name: "scout"
    description: "Read-only: lookup"
    model: "haiku"
    effort: "medium"
    maxTurns: 6
    disallowedTools: "Bash, Edit, Write, NotebookEdit, Agent"
    ---
    Stay bounded.
  ''; true
