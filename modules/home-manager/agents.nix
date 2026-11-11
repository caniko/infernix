# Harness-neutral subagent definitions rendered to OpenCode and Claude Code.
# OpenCode's permission vocabulary is canonical; Claude Code receives only
# agents that declare a Claude model or use a Claude route.
#
# Routes name the harness and model that run a role. An agent with a route is
# rendered only to that route's harness, so a role routed elsewhere can never
# fall back to a native agent pinned to another model.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.agents;
  routes = config.services.infernix.agentRoutes;
  claudeEfforts = ["low" "medium" "high" "xhigh" "max"];

  # Claude Code tools denied for each denied OpenCode permission. A denial with
  # no entry here fails evaluation rather than silently widening Claude access.
  claudeDeniedTools = {
    bash = ["Bash"];
    read = ["Read"];
    glob = ["Glob"];
    grep = ["Grep"];
    skill = ["Skill"];
    edit = ["Edit" "Write" "NotebookEdit"];
    task = ["Agent"];
    webfetch = ["WebFetch"];
    websearch = ["WebSearch"];
    question = ["AskUserQuestion"];
  };

  routeOf = agent:
    if agent.route == null
    then null
    else routes.${agent.route} or null;

  # The harness that runs a routed agent; null for unrouted agents.
  routedTo = agent: let
    route = routeOf agent;
  in
    if route == null
    then null
    else route.harness;

  inOpencode = agent: agent.route == null || routedTo agent == "opencode";

  # Model and effort for Claude Code, or null when the agent is not rendered there.
  claudeModel = agent: let
    route = routeOf agent;
  in
    if agent.route != null
    then
      if routedTo agent == "claude"
      then {
        inherit (route) model;
        effort = route.variant;
      }
      else null
    else if agent.claude != null && agent.claude.model != null
    then {inherit (agent.claude) model effort;}
    else null;

  toOpencode = _: agent: let
    route = routeOf agent;
  in
    {
      inherit (agent) description mode permission;
    }
    // lib.optionalAttrs (agent.steps != null) {inherit (agent) steps;}
    // lib.optionalAttrs (agent.prompt != null) {inherit (agent) prompt;}
    // lib.optionalAttrs (route != null) ({inherit (route) model;} // lib.optionalAttrs (route.variant != null) {inherit (route) variant;})
    // agent.opencode;

  # A Claude name equal to a built-in's (e.g. Explore) replaces that built-in.
  claudeName = name: agent:
    if agent.claude != null && agent.claude.name != null
    then agent.claude.name
    else name;

  # JSON strings are valid YAML double-quoted scalars.
  toClaude = name: agent: let
    denied = lib.concatLists (lib.mapAttrsToList (perm: action:
      lib.optionals (action == "deny") claudeDeniedTools.${perm})
    agent.permission);
    field = key: value: "${key}: ${builtins.toJSON value}";
    inherit (claudeModel agent) model effort;
  in
    lib.concatStringsSep "\n" (
      [
        "---"
        (field "name" (claudeName name agent))
        (field "description" agent.description)
        (field "model" model)
      ]
      ++ lib.optional (effort != null) (field "effort" effort)
      ++ lib.optional (agent.steps != null) (field "maxTurns" agent.steps)
      ++ lib.optional (denied != []) (field "disallowedTools" (lib.concatStringsSep ", " denied))
      ++ lib.optional (agent.claude != null && agent.claude.omitClaudeMd) (field "omitClaudeMd" true)
      ++ ["---" (lib.optionalString (agent.prompt != null) agent.prompt) ""]
    );

  # A route is the agent's only model source.
  routeAssertions = name: agent: let
    route = routeOf agent;
    where = "services.infernix.agents.${name}";
  in
    lib.optionals (agent.route != null) [
      {
        assertion = route != null;
        message = "${where}: route '${agent.route}' is not defined in services.infernix.agentRoutes.";
      }
      {
        assertion = !(agent.opencode ? model || agent.opencode ? variant);
        message = "${where}: set the model through its route, not opencode.model or opencode.variant.";
      }
      {
        assertion = agent.claude == null || (routedTo agent == "claude" && agent.claude.model == null && agent.claude.effort == null);
        message = "${where}: a routed agent takes its Claude model from the route; claude may only set name or omitClaudeMd, and only on a claude route.";
      }
    ]
    ++ lib.optional (agent.route == null && agent.claude != null) {
      assertion = agent.claude.model != null;
      message = "${where}: claude needs a model or a claude route.";
    };
in {
  options.services.infernix.agentRoutes = mkOption {
    default = {};
    description = "Named subagent routes. Each names the harness and model that run a role; agents select one with `route`.";
    type = types.attrsOf (types.submodule {
      options = {
        harness = mkOption {
          type = types.enum ["opencode" "claude"];
          description = "Harness that runs the route natively.";
        };
        model = mkOption {
          type = types.str;
          description = "Model in that harness: an OpenCode provider/model ID, or a Claude alias or full ID.";
        };
        variant = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "OpenCode variant, or Claude Code effort for a claude route.";
        };
        description = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "The work this route is for.";
        };
      };
    });
  };

  options.services.infernix.agents = mkOption {
    default = {};
    description = "Subagents defined once and rendered to OpenCode and, when a Claude model or route is set, Claude Code.";
    type = types.attrsOf (types.submodule {
      options = {
        description = mkOption {type = types.str;};
        prompt = mkOption {
          type = types.nullOr types.lines;
          default = null;
        };
        mode = mkOption {
          type = types.enum ["subagent" "primary" "all"];
          default = "subagent";
        };
        steps = mkOption {
          type = types.nullOr types.ints.positive;
          default = null;
          description = "OpenCode steps; Claude Code maxTurns.";
        };
        permission = mkOption {
          type = types.attrsOf (types.enum ["allow" "ask" "deny"]);
          default = {};
          description = "OpenCode permissions; denials map to Claude Code disallowedTools.";
        };
        route = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Name in services.infernix.agentRoutes. The agent is rendered only to that route's harness, with the route's model.";
        };
        opencode = mkOption {
          type = types.attrs;
          default = {};
          description = "Merged verbatim into the OpenCode agent (model, variant, ...).";
        };
        claude = mkOption {
          default = null;
          description = "Claude Code rendering; null keeps an unrouted agent OpenCode-only.";
          type = types.nullOr (types.submodule {
            options = {
              name = mkOption {
                type = types.nullOr types.str;
                default = null;
                description = "Claude Code agent name; defaults to the attribute name.";
              };
              omitClaudeMd = mkOption {
                type = types.bool;
                default = false;
                description = "Run without the user, project and local CLAUDE.md files.";
              };
              model = mkOption {
                type = types.nullOr types.str;
                default = null;
                description = "Claude model alias (haiku, sonnet, opus, fable, inherit) or full ID. Required unless the agent has a route.";
              };
              effort = mkOption {
                type = types.nullOr (types.enum claudeEfforts);
                default = null;
              };
            };
          });
        };
      };
    });
  };

  config = {
    assertions =
      lib.concatLists (lib.mapAttrsToList (name: agent:
        routeAssertions name agent
        ++ lib.optionals (claudeModel agent != null) (lib.mapAttrsToList (perm: action: {
            assertion = action != "deny" || claudeDeniedTools ? ${perm};
            message = "services.infernix.agents.${name}: denied permission '${perm}' has no Claude Code tool mapping.";
          })
          agent.permission))
      cfg)
      ++ lib.mapAttrsToList (name: route: {
        assertion = route.harness != "claude" || route.variant == null || lib.elem route.variant claudeEfforts;
        message = "services.infernix.agentRoutes.${name}: a claude route's variant is a Claude Code effort (${lib.concatStringsSep ", " claudeEfforts}).";
      })
      routes;
    programs.opencode.settings.agent = lib.mkIf (cfg != {}) (lib.mapAttrs toOpencode (lib.filterAttrs (_: inOpencode) cfg));
    home.file = lib.mkIf (cfg != {}) (lib.mapAttrs' (name: agent:
      lib.nameValuePair ".claude/agents/${claudeName name agent}.md" {text = toClaude name agent;})
    (lib.filterAttrs (_: agent: claudeModel agent != null) cfg));
  };
}
