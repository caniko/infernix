# Harness-neutral subagent definitions rendered to OpenCode and Claude Code.
# OpenCode's permission vocabulary is canonical; Claude Code receives only
# agents that declare a Claude model.
{
  config,
  lib,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.agents;

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

  toOpencode = _: agent:
    {
      inherit (agent) description mode permission;
    }
    // lib.optionalAttrs (agent.steps != null) {inherit (agent) steps;}
    // lib.optionalAttrs (agent.prompt != null) {inherit (agent) prompt;}
    // agent.opencode;

  # JSON strings are valid YAML double-quoted scalars.
  toClaude = name: agent: let
    denied = lib.concatLists (lib.mapAttrsToList (perm: action:
      lib.optionals (action == "deny") claudeDeniedTools.${perm})
    agent.permission);
    field = key: value: "${key}: ${builtins.toJSON value}";
  in
    lib.concatStringsSep "\n" (
      [
        "---"
        (field "name" name)
        (field "description" agent.description)
        (field "model" agent.claude.model)
      ]
      ++ lib.optional (agent.claude.effort != null) (field "effort" agent.claude.effort)
      ++ lib.optional (agent.steps != null) (field "maxTurns" agent.steps)
      ++ lib.optional (denied != []) (field "disallowedTools" (lib.concatStringsSep ", " denied))
      ++ ["---" (lib.optionalString (agent.prompt != null) agent.prompt) ""]
    );
in {
  options.services.infernix.agents = mkOption {
    default = {};
    description = "Subagents defined once and rendered to OpenCode and, when a Claude model is set, Claude Code.";
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
        opencode = mkOption {
          type = types.attrs;
          default = {};
          description = "Merged verbatim into the OpenCode agent (model, variant, ...).";
        };
        claude = mkOption {
          default = null;
          description = "Claude Code rendering; null keeps the agent OpenCode-only.";
          type = types.nullOr (types.submodule {
            options = {
              model = mkOption {
                type = types.str;
                description = "Claude model alias (haiku, sonnet, opus, fable, inherit) or full ID.";
              };
              effort = mkOption {
                type = types.nullOr (types.enum ["low" "medium" "high" "xhigh" "max"]);
                default = null;
              };
            };
          });
        };
      };
    });
  };

  config = lib.mkIf (cfg != {}) {
    assertions = lib.concatLists (lib.mapAttrsToList (name: agent:
      lib.optionals (agent.claude != null) (lib.mapAttrsToList (perm: action: {
          assertion = action != "deny" || claudeDeniedTools ? ${perm};
          message = "services.infernix.agents.${name}: denied permission '${perm}' has no Claude Code tool mapping.";
        })
        agent.permission))
    cfg);
    programs.opencode.settings.agent = lib.mapAttrs toOpencode cfg;
    home.file = lib.mapAttrs' (name: agent:
      lib.nameValuePair ".claude/agents/${name}.md" {text = toClaude name agent;})
    (lib.filterAttrs (_: agent: agent.claude != null) cfg);
  };
}
