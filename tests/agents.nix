{
  pkgs,
  homeManager,
}: let
  evalIn = extraModules: routes: agents:
    (homeManager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules =
        extraModules
        ++ [
        ../modules/home-manager/agents.nix
        # A second path import must deduplicate (homeModules.agents next to default).
        ../modules/home-manager/agents.nix
        {
          home = {
            username = "tester";
            homeDirectory = "/home/tester";
            stateVersion = "24.11";
          };
          services.infernix = {
            inherit agents;
            agentRoutes = routes;
          };
        }
      ];
    }).config;
  evalWith = evalIn [];
  # homeModules.opencode sets models for its built-in general, build and plan agents.
  withOpencode = evalIn [
    ../modules/home-manager/model-providers.nix
    ../modules/home-manager/opencode.nix
  ];
  eval = evalWith {};
  routes = {
    review = {
      harness = "opencode";
      model = "openai/gpt-6.1-sol";
      variant = "xhigh";
    };
    lead = {
      harness = "claude";
      model = "opus";
      variant = "high";
    };
  };
  # Home Manager throws "Failed assertions" when its config is read.
  failsAssertions = config: !(builtins.tryEval (builtins.deepSeq config.assertions config.home.file)).success;
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
    explore = {
      description = "Search";
      claude = {
        name = "Explore";
        model = "sonnet";
        omitClaudeMd = true;
      };
    };
  };
  # A denial Claude Code cannot express must fail evaluation, not widen access.
  unmapped = eval {
    unmapped = {
      description = "Denies a permission Claude cannot express";
      permission.doom_loop = "deny";
      claude.model = "haiku";
    };
  };
  routed = evalWith routes {
    # Routed to OpenCode: no native Claude agent may exist for it.
    visual = {
      description = "Vision";
      route = "review";
      permission.edit = "deny";
    };
    planner = {
      description = "Plan";
      route = "lead";
      steps = 4;
      claude.name = "Plan";
    };
  };
  composed = withOpencode routes {
    general = {
      description = "General";
      route = "review";
    };
    plan = {
      description = "Plan";
      route = "lead";
    };
  };
  coc = composed.programs.opencode.settings.agent;
  oc = config.programs.opencode.settings.agent;
  roc = routed.programs.opencode.settings.agent;
in
  assert oc.scout.model == "openai/sol" && oc.scout.steps == 6 && oc.scout.mode == "subagent";
  assert oc.scout.permission.doom_loop == "ask";
  assert oc.title
  == {
    description = "OpenCode-only";
    mode = "subagent";
    permission = {};
  };
  assert !(failsAssertions config);
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
  '';
  assert !(config.home.file ? ".claude/agents/explore.md");
  assert config.home.file.".claude/agents/Explore.md".text
  == ''
    ---
    name: "Explore"
    description: "Search"
    model: "sonnet"
    omitClaudeMd: true
    ---

  '';
  # Routes set the model in their own harness only.
  assert !(failsAssertions routed);
  assert roc.visual.model == "openai/gpt-6.1-sol" && roc.visual.variant == "xhigh";
  assert !(routed.home.file ? ".claude/agents/visual.md");
  assert !(roc ? planner);
  assert routed.home.file.".claude/agents/Plan.md".text
  == ''
    ---
    name: "Plan"
    description: "Plan"
    model: "opus"
    effort: "high"
    maxTurns: 4
    ---

  '';
  # A route is the only model source, and must exist.
  assert failsAssertions (evalWith routes {
    x = {
      description = "Unknown route";
      route = "missing";
    };
  });
  assert failsAssertions (evalWith routes {
    x = {
      description = "Second model source";
      route = "review";
      opencode.model = "openai/sol";
    };
  });
  assert failsAssertions (evalWith routes {
    x = {
      description = "Native pin beside an OpenCode route";
      route = "review";
      claude.model = "fable";
    };
  });
  assert failsAssertions (eval {
    x = {
      description = "Claude without a model";
      claude.name = "X";
    };
  });
  # Routes are validated even before any agent uses them.
  assert failsAssertions (evalWith {
      bad = {
        harness = "claude";
        model = "opus";
        variant = "turbo";
      };
    } {});
  # Routed built-ins replace homeModules.opencode's defaults instead of conflicting.
  assert !(failsAssertions composed);
  assert coc.general.model == "openai/gpt-6.1-sol" && coc.general.variant == "xhigh";
  assert !(coc ? plan);
  assert coc ? build; true
