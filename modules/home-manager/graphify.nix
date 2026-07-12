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
  openaiPython = pkgs.python312.withPackages (pythonPackages: [
    pythonPackages.mcp
    pythonPackages.openai
    pythonPackages.tiktoken
  ]);
  graphifyRuntimePackage =
    if graphifyPackage == null
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
    in ''
      ${refreshSharedRoots}
      echo "infernix graphify: ${lib.escapeShellArg command}"
      ${command}
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
    if !cfg.enable || endpoint == null || model == null || baseUrl == null
    then {}
    else {
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
        consumer may override this with a package carrying optional semantic
        extraction dependencies, while Infernix retains ownership of the
        registration boundary.
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
        assertion = endpoints ? ${cfg.endpoint};
        message = "services.infernix.graphify.endpoint refers to '${cfg.endpoint}', which is not declared in services.infernix.endpoints.";
      }
      {
        assertion = endpoint == null || endpoint.type == "llama-swap";
        message = "services.infernix.graphify.endpoint '${cfg.endpoint}' must be a llama-swap endpoint (OpenAI-compatible).";
      }
      {
        assertion = endpoint == null || endpoint.models ? ${cfg.model};
        message = "services.infernix.graphify.model '${cfg.model}' is not defined on endpoint '${cfg.endpoint}'.";
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
