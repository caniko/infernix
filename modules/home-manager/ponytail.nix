{
  config,
  infernixPonytail ? null,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.ponytail;
  ponytailRevision = "16f29800fd2681bdf24f3eb4ccffe38be3baec6b";
  defaultHarnesses = [
    "agents"
    "aider"
    "amp"
    "antigravity"
    "claude"
    "cline"
    "claw"
    "codebuddy"
    "codewhale"
    "codex"
    "copilot"
    "copilot-cli"
    "cursor"
    "devin"
    "droid"
    "gemini"
    "hermes"
    "jules"
    "junie"
    "kilo"
    "kiro"
    "kimi"
    "opencode"
    "pi"
    "qoder"
    "swival"
    "trae"
    "trae-cn"
    "vscode"
    "windsurf"
    "zed"
  ];
  nativeHarnesses = [
    "claude"
    "codex"
    "copilot-cli"
    "devin"
    "gemini"
    "hermes"
    "opencode"
    "pi"
    "qoder"
  ];
  projectOnlyHarnesses = [
    "aider"
    "antigravity"
    "cline"
    "codebuddy"
    "codewhale"
    "copilot"
    "cursor"
    "jules"
    "junie"
    "trae"
    "trae-cn"
    "vscode"
    "windsurf"
    "zed"
  ];
  package = if cfg.package != null then cfg.package else infernixPonytail;
  runtimeDir = cfg.runtimeDir;
  mode = types.enum [ "off" "lite" "full" "ultra" ];
  adapterStatus =
    lib.genAttrs cfg.harnesses (harness: {
      mode = if builtins.elem harness nativeHarnesses then "native" else "instruction";
      scope = if builtins.elem harness projectOnlyHarnesses then "project-only" else "global";
      source =
        if builtins.elem harness nativeHarnesses
        then "pinned Ponytail native adapter"
        else "pinned Ponytail rules and skills";
    });
  beginMarker = "<!-- infernix-ponytail: begin -->";
  endMarker = "<!-- infernix-ponytail: end -->";
  jq = "${pkgs.jq}/bin/jq";
  node = "${pkgs.nodejs}/bin/node";
  runtimeStore = if package == null then "/nonexistent" else toString package;
  jsonHookScript = ''
    merge_hook_json() {
      hook_file="$1"
      include_subagent="$2"
      hook_dir="$(dirname "$hook_file")"
      mkdir -p "$hook_dir"
      if [ -e "$hook_file" ] && [ ! -f "$hook_file" ]; then
        echo "infernix ponytail: refusing non-file JSON target $hook_file" >&2
        exit 1
      fi
      if [ ! -e "$hook_file" ]; then
        printf '{}\n' > "$hook_file"
      fi
      if ! ${jq} -e 'type == "object" and (.hooks == null or (.hooks | type) == "object")' "$hook_file" >/dev/null; then
        echo "infernix ponytail: invalid hook configuration $hook_file" >&2
        exit 1
      fi
      additions="$(${jq} -n \
        --arg activate "$runtime_dir/hooks/ponytail-activate.js" \
        --arg tracker "$runtime_dir/hooks/ponytail-mode-tracker.js" \
        --arg subagent "$runtime_dir/hooks/ponytail-subagent.js" \
        --arg node "${node}" \
        --argjson include_subagent "$include_subagent" '
          def command($script): ($node + " " + ($script | @sh));
          {
            SessionStart: [{
              matcher: "startup|resume|clear|compact",
              hooks: [{type: "command", command: command($activate)}]
            }],
            UserPromptSubmit: [{
              hooks: [{type: "command", command: command($tracker)}]
            }]
          }
          + (if $include_subagent then {
            SubagentStart: [{
              hooks: [{type: "command", command: command($subagent)}]
            }]
          } else {} end)')"
      tmp="$(mktemp "$hook_file.XXXXXX")"
      if ! ${jq} --argjson additions "$additions" '
        def append_unique($old; $new):
          reduce $new[] as $item ($old; if any(.[]; . == $item) then . else . + [$item] end);
        .hooks = (.hooks // {})
        | reduce ($additions | to_entries[]) as $entry (.;
            .hooks[$entry.key] = append_unique((.hooks[$entry.key] // []); $entry.value))
      ' "$hook_file" > "$tmp"; then
        rm -f "$tmp"
        echo "infernix ponytail: failed to merge $hook_file" >&2
        exit 1
      fi
      mv "$tmp" "$hook_file"
    }
  '';
  jsonPluginScript = ''
    merge_opencode_json() {
      opencode_file="$1"
      opencode_dir="$(dirname "$opencode_file")"
      mkdir -p "$opencode_dir"
      if [ -e "$opencode_file" ] && [ ! -f "$opencode_file" ]; then
        echo "infernix ponytail: refusing non-file OpenCode target $opencode_file" >&2
        exit 1
      fi
      if [ ! -e "$opencode_file" ]; then
        printf '{}\n' > "$opencode_file"
      fi
      if ! ${jq} -e 'type == "object" and (.plugin == null or (.plugin | type) == "array")' "$opencode_file" >/dev/null; then
        echo "infernix ponytail: invalid OpenCode configuration $opencode_file" >&2
        exit 1
      fi
      tmp="$(mktemp "$opencode_file.XXXXXX")"
      if ! ${jq} --arg plugin "$runtime_dir/.opencode/plugins/ponytail.mjs" '
        .plugin = (((.plugin // []) + [$plugin]) | unique)
      ' "$opencode_file" > "$tmp"; then
        rm -f "$tmp"
        echo "infernix ponytail: failed to merge $opencode_file" >&2
        exit 1
      fi
      mv "$tmp" "$opencode_file"
    }
  '';
  managedBlockScript = ''
    managed_block() {
      target="$1"
      source="$2"
      target_dir="$(dirname "$target")"
      mkdir -p "$target_dir"
      if [ -e "$target" ] && [ ! -f "$target" ]; then
        echo "infernix ponytail: refusing non-file target $target" >&2
        exit 1
      fi
      if [ ! -e "$source" ]; then
        echo "infernix ponytail: missing packaged adapter source $source" >&2
        exit 1
      fi
      stripped="$(mktemp "$target.XXXXXX")"
      if [ -f "$target" ]; then
        ${pkgs.gawk}/bin/awk -v begin='${beginMarker}' -v end='${endMarker}' '
          $0 == begin {inside=1; next}
          $0 == end {inside=0; next}
          !inside {print}
        ' "$target" > "$stripped"
      else
        : > "$stripped"
      fi
      rendered="$(mktemp "$target.XXXXXX")"
      cat "$stripped" > "$rendered"
      if [ -s "$stripped" ]; then printf '\n' >> "$rendered"; fi
      printf '%s\n' '${beginMarker}' >> "$rendered"
      cat "$source" >> "$rendered"
      printf '\n%s\n' '${endMarker}' >> "$rendered"
      mv "$rendered" "$target"
      rm -f "$stripped"
    }
  '';
  managedLinkScript = ''
    managed_link() {
      source="$1"
      target="$2"
      if [ ! -e "$source" ]; then
        echo "infernix ponytail: missing packaged adapter source $source" >&2
        exit 1
      fi
      mkdir -p "$(dirname "$target")"
      if [ -e "$target" ] && [ ! -L "$target" ]; then
        echo "infernix ponytail: refusing to replace user path $target" >&2
        exit 1
      fi
      if [ -L "$target" ]; then
        current="$(readlink "$target")"
        case "$current" in
          "$runtime_dir"|"$runtime_dir"/*) ;;
          *)
            echo "infernix ponytail: refusing to replace unrelated symlink $target" >&2
            exit 1
            ;;
        esac
      fi
      link_tmp="$(mktemp -u "$target.XXXXXX")"
      ln -s "$source" "$link_tmp"
      mv -Tf "$link_tmp" "$target"
    }
  '';
in {
  options.services.infernix.ponytail = {
    enable = mkEnableOption "Ponytail guidance and harness integrations";

    package = mkOption {
      type = types.nullOr types.package;
      default = infernixPonytail;
      description = "Pinned Ponytail runtime package.";
    };

    harnesses = mkOption {
      type = types.listOf (types.enum defaultHarnesses);
      default = defaultHarnesses;
      description = "Ponytail harnesses to wire during Home Manager activation.";
    };

    runtimeDir = mkOption {
      type = types.path;
      default = "${config.home.homeDirectory}/.local/share/infernix/ponytail";
      description = "Stable user-owned path used by native Ponytail adapters.";
    };

    defaultMode = mkOption {
      type = types.nullOr mode;
      default = null;
      description = "Optional persisted Ponytail mode; null preserves user configuration.";
    };

    subagentMatcher = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Optional PONYTAIL_SUBAGENT_MATCHER value.";
    };

    registeredHarnesses = mkOption {
      type = types.listOf types.str;
      readOnly = true;
      description = "Harnesses wired by Infernix.";
    };

    adapterStatus = mkOption {
      type = types.attrs;
      readOnly = true;
      description = "Read-only adapter mode and scope metadata.";
    };
  };

  config = {
    services.infernix.ponytail.registeredHarnesses = cfg.harnesses;
    services.infernix.ponytail.adapterStatus = adapterStatus;

    assertions = lib.optionals cfg.enable [
      {
        assertion = package != null;
        message = "services.infernix.ponytail.package must resolve when Ponytail is enabled.";
      }
      {
        assertion = lib.unique cfg.harnesses == cfg.harnesses;
        message = "services.infernix.ponytail.harnesses must not contain duplicates.";
      }
    ];

    home.packages = lib.mkIf cfg.enable [pkgs.nodejs];
    home.sessionVariables = lib.mkIf (cfg.enable && cfg.subagentMatcher != null) {
      PONYTAIL_SUBAGENT_MATCHER = cfg.subagentMatcher;
    };

    home.activation.infernixPonytail = lib.mkIf cfg.enable (lib.hm.dag.entryAfter ["writeBoundary"] ''
      set -eu
      runtime_store=${lib.escapeShellArg runtimeStore}
      runtime_dir=${lib.escapeShellArg (toString runtimeDir)}
      runtime_parent="$(dirname "$runtime_dir")"
      runtime_version="$runtime_parent/ponytail-${ponytailRevision}"
      mkdir -p "$runtime_parent"
      if [ -e "$runtime_dir" ] && [ ! -L "$runtime_dir" ]; then
        echo "infernix ponytail: refusing to replace non-managed runtime path $runtime_dir" >&2
        exit 1
      fi
      if [ ! -e "$runtime_version" ]; then
        runtime_tmp="$(mktemp -d "$runtime_parent/.ponytail.XXXXXX")"
        cp -a "$runtime_store/." "$runtime_tmp/"
        chmod -R u+w "$runtime_tmp"
        printf '%s\n' '${ponytailRevision}' > "$runtime_tmp/.infernix-ponytail-revision"
        mv "$runtime_tmp" "$runtime_version"
      fi
      runtime_link="$(mktemp -u "$runtime_parent/.ponytail-link.XXXXXX")"
      ln -s "$runtime_version" "$runtime_link"
      mv -Tf "$runtime_link" "$runtime_dir"

      ${managedBlockScript}
      ${managedLinkScript}
      ${jsonHookScript}
      ${jsonPluginScript}

      managed_block "$HOME/AGENTS.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.agents/rules/ponytail.md" "$runtime_dir/.agents/rules/ponytail.md"
      managed_block "$HOME/.codex/AGENTS.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.config/opencode/AGENTS.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.claude/CLAUDE.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.copilot/copilot-instructions.md" "$runtime_dir/.github/copilot-instructions.md"
      managed_block "$HOME/.config/amp/AGENTS.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.config/swival/AGENTS.md" "$runtime_dir/AGENTS.md"
      managed_block "$HOME/.cursor/rules/ponytail.mdc" "$runtime_dir/.cursor/rules/ponytail.mdc"
      managed_block "$HOME/.windsurf/rules/ponytail.md" "$runtime_dir/.windsurf/rules/ponytail.md"
      managed_block "$HOME/.clinerules/ponytail.md" "$runtime_dir/.clinerules/ponytail.md"
      managed_block "$HOME/.kiro/steering/ponytail.md" "$runtime_dir/.kiro/steering/ponytail.md"
      managed_block "$HOME/.qoder/rules/ponytail.md" "$runtime_dir/.qoder/rules/ponytail.md"

      merge_hook_json "$HOME/.codex/hooks.json" false
      merge_hook_json "$HOME/.claude/settings.json" true

      opencode_config_count=0
      for opencode_config in "$HOME/.config/opencode/opencode.json" "$HOME/.opencode/opencode.json"; do
        if [ -f "$opencode_config" ]; then
          merge_opencode_json "$opencode_config"
          opencode_config_count=$((opencode_config_count + 1))
        fi
      done
      if [ "$opencode_config_count" -eq 0 ]; then
        merge_opencode_json "$HOME/.config/opencode/opencode.json"
      fi

      managed_link "$runtime_dir/pi-extension" "$HOME/.pi/agent/extensions/ponytail"
      managed_link "$runtime_dir" "$HOME/.gemini/extensions/ponytail"
      managed_link "$runtime_dir/.openclaw/skills/ponytail" "$HOME/.openclaw/skills/ponytail"
      managed_link "$runtime_dir/.devin-plugin" "$HOME/.devin/plugins/ponytail"
      managed_link "$runtime_dir/.github/plugin" "$HOME/.copilot/plugins/ponytail"
      managed_link "$runtime_dir/.codex-plugin" "$HOME/.codex/plugins/ponytail"
      managed_link "$runtime_dir/.claude-plugin" "$HOME/.claude/plugins/ponytail"

      if command -v hermes >/dev/null 2>&1; then
        managed_link "$runtime_dir" "$HOME/.hermes/plugins/ponytail"
        hermes plugins enable ponytail --no-allow-tool-override >/dev/null 2>&1 || {
          echo "infernix ponytail: Hermes plugin could not be enabled; rules remain installed" >&2
        }
      fi

      ${lib.optionalString (cfg.defaultMode != null) ''
        mkdir -p "$HOME/.config/ponytail"
        mode_tmp="$(mktemp "$HOME/.config/ponytail/config.json.XXXXXX")"
        if [ -f "$HOME/.config/ponytail/config.json" ]; then
          ${jq} --arg mode ${lib.escapeShellArg cfg.defaultMode} '.defaultMode = $mode' "$HOME/.config/ponytail/config.json" > "$mode_tmp"
        else
          ${jq} -n --arg mode ${lib.escapeShellArg cfg.defaultMode} '{defaultMode: $mode}' > "$mode_tmp"
        fi
        mv "$mode_tmp" "$HOME/.config/ponytail/config.json"
      ''}

    '');
  };
}
