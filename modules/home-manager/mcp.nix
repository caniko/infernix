{ config
, lib
, pkgs
, infernixOpenPencil ? null
, ...
}:
let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.openpencil;
  manifest =
    if infernixOpenPencil != null
      && infernixOpenPencil ? lib
      && infernixOpenPencil.lib ? integrationManifest
    then infernixOpenPencil.lib.integrationManifest.integration
    else null;
  systemPackages =
    if manifest != null && infernixOpenPencil.packages ? ${pkgs.system}
    then infernixOpenPencil.packages.${pkgs.system}
    else { };
  defaultPackage =
    if manifest != null && systemPackages ? ${manifest.packages.prebuiltRuntime}
    then systemPackages.${manifest.packages.prebuiltRuntime}
    else null;
  transport =
    if cfg.transport == "http"
    then {
      type = "http";
      url = "http://127.0.0.1:${toString cfg.live.port}/mcp";
    }
    else {
      command = "${cfg.package}/bin/${manifest.executables.desktop}";
      args = [ "--mcp" cfg.document ];
    };
  server = transport // { inherit (cfg) environment; };
  harnesses = lib.genAttrs cfg.harnesses (name:
    if manifest != null && manifest.harnesses ? ${name}
    then manifest.harnesses.${name}
    else null);
  jsonHarnesses = lib.filterAttrs (_: adapter: adapter != null && adapter.format == "json") harnesses;
  tomlHarnesses = lib.filterAttrs (_: adapter: adapter != null && adapter.format == "toml") harnesses;
  jsonPlan = builtins.toJSON (lib.mapAttrs (_: adapter: {
    path = adapter.configPath;
    key = adapter.serverKey;
    value = server;
  }) jsonHarnesses);
  tomlPlan = builtins.toJSON (lib.mapAttrs (_: adapter: {
    path = adapter.configPath;
    key = adapter.serverKey;
    value = server;
  }) tomlHarnesses);
  reconcile = pkgs.writeText "infernix-openpencil-reconcile.py" ''
    import json, os, pathlib, tempfile
    from tomlkit import dumps, load

    def atomic_write(path, text):
        path = pathlib.Path(os.path.expanduser(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".infernix-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)

    def json_merge(item):
        path = os.path.expanduser(item["path"])
        try:
            with open(path) as handle: document = json.load(handle)
        except FileNotFoundError: document = {}
        document.setdefault("mcpServers", {})[item["key"]] = item["value"]
        atomic_write(path, json.dumps(document, indent=2, sort_keys=True) + "\n")

    def toml_merge(item):
        path = os.path.expanduser(item["path"])
        try:
            with open(path) as handle: document = load(handle)
        except FileNotFoundError: document = {}
        table = document.setdefault("mcp_servers", {})
        table[item["key"]] = item["value"]
        atomic_write(path, dumps(document))

    mode = os.environ["INFERNIX_OPENPENCIL_MODE"]
    plan = json.loads(os.environ["INFERNIX_OPENPENCIL_PLAN"])
    for item in plan.values():
        (toml_merge if mode == "toml" else json_merge)(item)
  '';
in {
  options.services.infernix.mcp = {
    servers = mkOption {
      type = types.attrsOf (types.submodule ({ ... }: {
        options = {
          transport = mkOption { type = types.enum [ "stdio" "http" ]; default = "stdio"; };
          command = mkOption { type = types.str; default = ""; };
          args = mkOption { type = types.listOf types.str; default = [ ]; };
          url = mkOption { type = types.nullOr types.str; default = null; };
          environment = mkOption { type = types.attrsOf types.str; default = { }; };
          harnesses = mkOption { type = types.listOf types.str; default = [ ]; };
        };
      }));
      default = { };
      description = "Declarative MCP server registry shared by downstream harnesses.";
    };
    resolvedServers = mkOption {
      type = types.attrsOf (types.attrsOf types.anything);
      readOnly = true;
      description = "Resolved MCP server descriptors for downstream modules.";
    };
  };

  options.services.infernix.openpencil = {
    enable = mkEnableOption "OpenPencil MCP integration";
    package = mkOption {
      type = types.nullOr types.package;
      default = defaultPackage;
      description = "OpenPencil runtime package selected from the locked integration manifest.";
    };
    transport = mkOption { type = types.enum [ "stdio" "http" ]; default = "stdio"; };
    document = mkOption {
      type = types.str;
      default = "${config.xdg.dataHome}/openpencil/agent.op";
    };
    live.port = mkOption { type = types.port; default = 3100; };
    environment = mkOption { type = types.attrsOf types.str; default = { }; };
    harnesses = mkOption {
      type = types.listOf types.str;
      default = [ "claude" "codex" "gemini" "opencode" "kiro" "copilot" "antigravity" "hermes" ];
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      { assertion = manifest != null; message = "services.infernix.openpencil requires an OpenPencil flake exposing lib.integrationManifest."; }
      { assertion = cfg.package != null; message = "services.infernix.openpencil package is unavailable for this system; set package explicitly or disable the integration."; }
    ];
    services.infernix.mcp.servers.openpencil = {
      transport = cfg.transport;
      command = "${cfg.package}/bin/${manifest.executables.desktop}";
      args = [ "--mcp" cfg.document ];
      url = lib.mkIf (cfg.transport == "http") "http://127.0.0.1:${toString cfg.live.port}/mcp";
      inherit (cfg) environment harnesses;
    };
    services.infernix.mcp.resolvedServers = config.services.infernix.mcp.servers;
    home.activation.infernixOpenPencil = lib.hm.dag.entryAfter [ "writeBoundary" ] ''
      install -d -m 0755 "$(dirname ${lib.escapeShellArg cfg.document})"
      if [ ! -e ${lib.escapeShellArg cfg.document} ]; then
        install -m 0644 "${cfg.package}/${manifest.documentTemplate}" ${lib.escapeShellArg cfg.document}
      fi
      INFERNIX_OPENPENCIL_MODE=json INFERNIX_OPENPENCIL_PLAN=${lib.escapeShellArg jsonPlan} \
        ${pkgs.python3.withPackages (p: [ p.tomlkit ])}/bin/python3 ${reconcile}
      INFERNIX_OPENPENCIL_MODE=toml INFERNIX_OPENPENCIL_PLAN=${lib.escapeShellArg tomlPlan} \
        ${pkgs.python3.withPackages (p: [ p.tomlkit ])}/bin/python3 ${reconcile}
    '';
    systemd.user.services.infernix-openpencil-mcp = lib.mkIf (cfg.transport == "http") {
      Unit.Description = "OpenPencil live MCP server";
      Service = {
        ExecStart = "${cfg.package}/bin/${manifest.executables.desktop} --live-mcp ${toString cfg.live.port} ${cfg.document}";
        Restart = "on-failure";
      };
      Install.WantedBy = [ "default.target" ];
    };
  };
}
