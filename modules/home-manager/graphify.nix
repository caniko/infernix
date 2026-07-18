# Expose Graphify through Infernix. Infernix owns the package revision,
# endpoint/model routing, and registration of the skill with every supported
# agent harness; Graphify only owns extraction and graph formats.
{
  config,
  infernixGraphify ? null,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.graphify;
  endpoints = config.services.infernix.endpoints;
  acpProviders = config.services.infernix.acp.resolvedProviders;
  system = pkgs.stdenv.hostPlatform.system;

  # These are Graphify's user-facing install targets. Windows-only variants
  # are intentionally not included: this module is evaluated on the Unix
  # Home Manager systems supported by Infernix, and installing the PowerShell
  # payload over the native Unix payload would make the skill unusable.
  allHarnesses = [
    "agents"
    "aider"
    "amp"
    "antigravity"
    "claude"
    "claw"
    "codebuddy"
    "codex"
    "copilot"
    "cursor"
    "devin"
    "droid"
    "gemini"
    "hermes"
    "kilo"
    "kiro"
    "kimi"
    "opencode"
    "pi"
    "trae"
    "trae-cn"
    "vscode"
  ];

  graphifyPackage =
    if infernixGraphify != null
    && infernixGraphify ? packages
    && infernixGraphify.packages ? ${system}
    then infernixGraphify.packages.${system}.default
    else null;
  graphifyFullPackage =
    if infernixGraphify != null
    && infernixGraphify ? packages
    && infernixGraphify.packages ? ${system}
    && infernixGraphify.packages.${system} ? full
    then infernixGraphify.packages.${system}.full
    else null;
  graphifyAcpPackage =
    if infernixGraphify != null
      && infernixGraphify ? packages
      && infernixGraphify.packages ? ${system}
      && infernixGraphify.packages.${system} ? acp
    then infernixGraphify.packages.${system}.acp
    else null;
  openaiPython = pkgs.python312.withPackages (pythonPackages: [
    pythonPackages.mcp
    pythonPackages.openai
    pythonPackages.tiktoken
  ]);
  graphifyRuntimePackage =
    if cfg.semanticBackend == "acp" && graphifyAcpPackage != null
    then graphifyAcpPackage
    else if graphifyFullPackage != null
    then graphifyFullPackage
    else if graphifyPackage == null
    then null
    else pkgs.symlinkJoin {
      name = "graphify-with-openai";
      paths = [graphifyPackage];
      nativeBuildInputs = [pkgs.makeWrapper];
      postBuild = ''
        wrapProgram "$out/bin/graphify" \
          --prefix PYTHONPATH : "${openaiPython}/${pkgs.python312.sitePackages}"
      '';
      meta = (graphifyPackage.meta or {}) // {mainProgram = "graphify";};
    };

  installCommand = harness:
    if builtins.elem harness ["kimi"]
    then "${lib.getExe cfg.package} install --platform ${harness}"
    else "${lib.getExe cfg.package} ${harness} install";
  registrationCommands =
    if cfg.package == null
    then []
    else map installCommand cfg.harnesses;

  acpProvider = acpProviders.${cfg.acp.provider} or null;
  acpConfigOptions =
    if acpProvider == null
    then { }
    else acpProvider.configOptions // cfg.acp.configOptions;
  graphifyInstallRoots = [
    ".agents/skills/graphify"
    ".aider/graphify"
    ".config/agents/skills/graphify"
    ".gemini/config/skills/graphify"
    ".claude/skills/graphify"
    ".openclaw/skills/graphify"
    ".codebuddy/skills/graphify"
    ".codex/skills/graphify"
    ".copilot/skills/graphify"
    ".config/devin/skills/graphify"
    ".factory/skills/graphify"
    ".gemini/skills/graphify"
    ".hermes/skills/graphify"
    ".config/kilo/skills/graphify"
    ".config/kilo/command/graphify.md"
    ".kiro/skills/graphify"
    ".kimi/skills/graphify"
    ".config/opencode/skills/graphify"
    ".pi/agent/skills/graphify"
    ".trae/skills/graphify"
    ".trae-cn/skills/graphify"
  ];
  makeGraphifyRootsWritable = lib.concatMapStringsSep "\n" (path: ''
    if [ -e "$HOME/${path}" ]; then
      ${pkgs.coreutils}/bin/chmod -R u+w -- "$HOME/${path}"
    fi
  '') graphifyInstallRoots;
  registrationScript = ''
    ${makeGraphifyRootsWritable}
    ${lib.concatMapStringsSep "\n" (harness: let
      command = installCommand harness;
      # Copilot CLI and VS Code Copilot Chat share ~/.copilot/skills. The
      # upstream installers copy read-only reference files from the Nix store,
      # so refresh modes immediately before the second shared install too.
      refreshSharedRoots = lib.optionalString (harness == "vscode") makeGraphifyRootsWritable;
      repairOpenCodePlugin = lib.optionalString (harness == "opencode") ''
        # graphify installs the global OpenCode plugin under ~/.opencode. In
        # that scope OpenCode resolves entries relative to the .opencode
        # directory itself, so the project-style plugins/graphify.js entry
        # becomes ~/.opencode/.opencode/plugins/graphify.js. Keep the repair
        # here at the Infernix harness boundary until the upstream installer
        # has a scope-aware global registration path.
        opencode_config="$HOME/.opencode/opencode.json"
        if [ -f "$opencode_config" ]; then
          opencode_config_tmp="$(mktemp "$opencode_config.XXXXXX")"
          if ${pkgs.jq}/bin/jq '
            if (.plugin | type) == "array" then
              .plugin |= map(if (. == ".opencode/plugins/graphify.js" or . == "plugins/graphify.js") then "./plugins/graphify.js" else . end) | .plugin |= unique
            else . end
          ' "$opencode_config" > "$opencode_config_tmp"; then
            ${pkgs.coreutils}/bin/mv "$opencode_config_tmp" "$opencode_config"
          else
            ${pkgs.coreutils}/bin/rm -f "$opencode_config_tmp"
            echo "infernix graphify: failed to repair $opencode_config" >&2
            exit 1
          fi
        fi
      '';
    in ''
      ${refreshSharedRoots}
      echo "infernix graphify: ${lib.escapeShellArg command}"
      ${command}
      ${repairOpenCodePlugin}
    '') cfg.harnesses}
  '';

  endpoint = endpoints.${cfg.endpoint} or null;
  model =
    if endpoint == null || cfg.model == null
    then null
    else endpoint.models.${cfg.model} or null;
  baseUrl =
    if endpoint == null || endpoint.url == null
    then null
    else if lib.hasSuffix "/v1" endpoint.url
    then endpoint.url
    else "${endpoint.url}/v1";
  generatedSettings =
    if !cfg.enable
    then {}
    else if cfg.semanticBackend == "acp" && acpProvider != null then {
      GRAPHIFY_SEMANTIC_BACKEND = "acp";
      GRAPHIFY_ACP_BIN = acpProvider.command;
      GRAPHIFY_ACP_ARGS_JSON = builtins.toJSON acpProvider.args;
      GRAPHIFY_ACP_CONFIG_JSON = builtins.toJSON acpConfigOptions;
      GRAPHIFY_ACP_MODEL = cfg.acp.model;
    } // acpProvider.environment
    else if endpoint == null || model == null || baseUrl == null then { }
    else {
      GRAPHIFY_SEMANTIC_BACKEND = "openai";
      OPENAI_BASE_URL = baseUrl;
      OPENAI_MODEL = model.name;
      # The OpenAI SDK requires a non-empty key even when the local endpoint
      # does not authenticate requests. Infernix never sends this placeholder
      # outside the configured local endpoint.
      OPENAI_API_KEY = "sk-infernix-local";
    };
in {
  options.services.infernix.graphify = {
    enable = mkEnableOption "Graphify semantic extraction through Infernix";

    semanticBackend = mkOption {
      type = types.enum [ "openai" "acp" ];
      default = "openai";
      description = "Semantic extraction transport. ACP uses the shared provider registry.";
    };

    acp = {
      provider = mkOption {
        type = types.str;
        default = "codex";
        description = "ACP provider name from services.infernix.acp.providers.";
      };
      model = mkOption {
        type = types.str;
        default = "gpt-5.5";
        description = "Model selected through ACP session configuration.";
      };
      configOptions = mkOption {
        type = types.attrsOf (types.oneOf [ types.str types.bool ]);
        default = { mode = "read-only"; };
        description = "Consumer ACP session settings merged over provider defaults.";
      };
    };

    endpoint = mkOption {
      type = types.str;
      default = "infernix-lb";
      description = "Endpoint name from services.infernix.endpoints.";
    };

    model = mkOption {
      type = types.str;
      default = "dsv4";
      description = "Model key from the selected Infernix endpoint.";
    };

    package = mkOption {
      type = types.nullOr types.package;
      default = graphifyRuntimePackage;
      description = ''
        Graphify package used for the command and harness registrations. A
        consumer may override this package, while Infernix retains ownership
        of the registration boundary. The upstream full package is preferred
        when available; the compatibility wrapper remains for pre-module
        Graphify revisions that expose only the lean default package.
      '';
    };

    harnesses = mkOption {
      type = types.listOf (types.enum allHarnesses);
      default = allHarnesses;
      description = ''
        Graphify harnesses registered during Home Manager activation. The
        default covers every Unix Graphify install target, including the
        generic Agent-Skills target and editor integrations.
      '';
    };

    registeredHarnesses = mkOption {
      type = types.listOf types.str;
      readOnly = true;
      description = "The harness registrations emitted by Infernix.";
    };

    registrationCommands = mkOption {
      type = types.listOf types.str;
      readOnly = true;
      description = "The idempotent Graphify installer commands emitted by Infernix.";
    };

    generatedSettings = mkOption {
      type = types.attrsOf types.str;
      readOnly = true;
      description = ''
        OpenAI-compatible environment values for Graphify semantic extraction.
        Consumers should pass these to Graphify rather than duplicating endpoint
        or model routing policy.
      '';
    };
  };

  config = {
    services.infernix.graphify.generatedSettings = generatedSettings;
    services.infernix.graphify.registeredHarnesses = cfg.harnesses;
    services.infernix.graphify.registrationCommands = registrationCommands;

    assertions = lib.optionals cfg.enable [
      {
        assertion = cfg.package != null;
        message = "services.infernix.graphify.package must resolve to a Graphify package when Graphify is enabled.";
      }
      {
        assertion = cfg.semanticBackend == "acp" || endpoints ? ${cfg.endpoint};
        message = "services.infernix.graphify.endpoint refers to '${cfg.endpoint}', which is not declared in services.infernix.endpoints.";
      }
      {
        assertion = cfg.semanticBackend == "acp" || endpoint == null || endpoint.type == "llama-swap";
        message = "services.infernix.graphify.endpoint '${cfg.endpoint}' must be a llama-swap endpoint (OpenAI-compatible).";
      }
      {
        assertion = cfg.semanticBackend == "acp" || endpoint == null || endpoint.models ? ${cfg.model};
        message = "services.infernix.graphify.model '${cfg.model}' is not defined on endpoint '${cfg.endpoint}'.";
      }
      {
        assertion = cfg.semanticBackend != "acp" || acpProvider != null;
        message = "services.infernix.graphify.acp.provider '${cfg.acp.provider}' is not declared in services.infernix.acp.providers.";
      }
    ];

    # Only install the package and mutate harness configuration when this
    # integration is explicitly enabled by the consuming home profile.
    # Graphify's own installers are idempotent and preserve unrelated config.
    # Keep this in Infernix so every harness receives one shared registration
    # policy rather than independent canix/user-module copies.
    #
    # The activation entry is added below through mkIf to avoid evaluating
    # lib.getExe on the null package in standalone fixtures.
    #
    # (The generated endpoint settings remain available even when disabled.)
    home.packages = lib.mkIf (cfg.enable && cfg.package != null) [cfg.package];
    home.activation.infernixGraphify = lib.mkIf cfg.enable (lib.hm.dag.entryAfter ["writeBoundary"] ''
      cd "$HOME"
      ${registrationScript}
    '');
  };
}
