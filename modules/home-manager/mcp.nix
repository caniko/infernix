{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkOption types;
  cfg = config.services.infernix.mcp;
  harnesses = config.services.infernix.harnesses;
  mcpHarnesses = lib.filterAttrs (_: value: value.adapters.mcp != null) harnesses;

  serverValue = server: let
    transport =
      if server.transport == "http"
      then {
        type = "http";
        url = server.url;
      }
      else {
        command = server.command;
        args = server.args;
      };
  in
    transport // {inherit (server) environment;};

  selectedHarnesses = server:
    if server.harnesses == null
    then builtins.attrNames mcpHarnesses
    else server.harnesses;

  adapterFor = name:
    if builtins.hasAttr name mcpHarnesses
    then mcpHarnesses.${name}.adapters.mcp
    else null;

  unsupportedFor = name:
    if builtins.hasAttr name harnesses
    then harnesses.${name}.unsupported
    else null;

  registrationPlan = lib.concatLists (lib.mapAttrsToList (
      serverName: server:
        lib.concatMap (harnessName: let
          adapter = adapterFor harnessName;
        in
          lib.optional (adapter != null) {
            harness = harnessName;
            format = adapter.format;
            path = adapter.configPath;
            root = adapter.root;
            key =
              if server.key != null
              then server.key
              else if adapter.serverKey != null
              then adapter.serverKey
              else serverName;
            value = serverValue server;
          })
        (selectedHarnesses server)
    )
    cfg.servers);

  validationAssertions =
    lib.concatLists (lib.mapAttrsToList (
        serverName: server: let
          selected = selectedHarnesses server;
        in
          (map (harnessName: {
              assertion = builtins.hasAttr harnessName harnesses;
              message = "services.infernix.mcp.servers.${serverName}.harnesses contains unknown harness '${harnessName}'.";
            })
            selected)
          ++ (map (harnessName: {
              assertion =
                (adapterFor harnessName)
                != null
                || unsupportedFor harnessName != null;
              message = "services.infernix.mcp.servers.${serverName}.harnesses selects '${harnessName}', but no supported MCP adapter is registered.";
            })
            selected)
      )
      cfg.servers)
    ++ lib.mapAttrsToList (serverName: server: {
      assertion = server.transport != "http" || server.url != null;
      message = "services.infernix.mcp.servers.${serverName}.url must be set for HTTP transport.";
    })
    cfg.servers;

  reconcile = pkgs.writeText "infernix-mcp-reconcile.py" ''
    import collections
    import json
    import os
    import pathlib
    import tempfile
    from collections.abc import MutableMapping
    from tomlkit import dumps, document, load


    def expand_path(value):
        path = pathlib.Path(os.path.expandvars(os.path.expanduser(value)))
        return path if path.is_absolute() else pathlib.Path.home() / path


    def atomic_write(path, text):
        path = expand_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".infernix-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


    def load_document(path, mode):
        path = expand_path(path)
        if not path.exists():
            return document() if mode == "toml" else {}
        with path.open() as handle:
            return load(handle) if mode == "toml" else json.load(handle)


    def table_at(document_value, root):
        table = document_value
        for segment in root:
            if not isinstance(table, MutableMapping):
                raise ValueError("MCP root contains a non-table at %s" % segment)
            if segment not in table:
                table[segment] = {}
            table = table[segment]
        if not isinstance(table, MutableMapping):
            raise ValueError("MCP root is not a table")
        return table


    def render(path, mode, document_value):
        if mode == "toml":
            return dumps(document_value)
        return json.dumps(document_value, indent=2, sort_keys=True) + "\n"


    status_path = expand_path(os.environ["INFERNIX_HARNESS_STATUS"])
    plan = json.loads(os.environ["INFERNIX_MCP_PLAN"])
    try:
        try:
            with status_path.open() as handle:
                status = json.load(handle)
        except FileNotFoundError:
            status = {"active": [], "failed": []}

        active = set(status.get("active", []))
        groups = collections.OrderedDict()
        for item in plan:
            if item["harness"] not in active:
                continue
            group = (item["format"], item["path"])
            groups.setdefault(group, []).append(item)

        rendered = []
        for (mode, path), items in groups.items():
            document_value = load_document(path, mode)
            for item in items:
                table = table_at(document_value, item["root"])
                table[item["key"]] = item["value"]
            rendered.append((path, mode, render(path, mode, document_value), items))

        configured = []
        for path, mode, text, items in rendered:
            atomic_write(path, text)
            configured.extend(item["harness"] for item in items)

        status["mcp"] = {
            "configured": sorted(set(configured)),
            "skipped": sorted(set(item["harness"] for item in plan) - set(configured)),
        }
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name("." + status_path.name + ".tmp")
        with temporary.open("w") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, status_path)
    except Exception as error:
        status.setdefault("failed", []).append({"kind": "mcp", "error": str(error)})
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name("." + status_path.name + ".tmp")
        with temporary.open("w") as handle:
            json.dump(status, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, status_path)
        raise
  '';
in {
  options.services.infernix.mcp = {
    servers = mkOption {
      type = types.attrsOf (types.submodule ({...}: {
        options = {
          transport = mkOption {
            type = types.enum ["stdio" "http"];
            default = "stdio";
          };
          command = mkOption {
            type = types.str;
            default = "";
          };
          args = mkOption {
            type = types.listOf types.str;
            default = [];
          };
          url = mkOption {
            type = types.nullOr types.str;
            default = null;
          };
          environment = mkOption {
            type = types.attrsOf types.str;
            default = {};
          };
          key = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Optional MCP key override; defaults to the adapter or server name.";
          };
          harnesses = mkOption {
            type = types.nullOr (types.listOf types.str);
            default = null;
            description = "Harness allowlist; null selects every detected registered adapter.";
          };
        };
      }));
      default = {};
      description = "Declarative MCP server registry shared by downstream harnesses.";
    };
    resolvedServers = mkOption {
      type = types.attrsOf (types.attrsOf types.anything);
      readOnly = true;
      description = "Resolved MCP server descriptors for downstream modules.";
    };
  };

  config = {
    assertions = validationAssertions;
    services.infernix.mcp.resolvedServers = cfg.servers;
    home.activation.infernixMcp = lib.mkIf (cfg.servers != {}) (lib.hm.dag.entryAfter ["infernixHarnessRegistry"] ''
      export PATH=${lib.escapeShellArg "${config.home.profileDirectory}/bin"}:$PATH
      INFERNIX_MCP_PLAN=${lib.escapeShellArg (builtins.toJSON registrationPlan)} \
        INFERNIX_HARNESS_STATUS=${lib.escapeShellArg config.services.infernix.harnessRegistry.statusFile} \
        ${pkgs.python3.withPackages (pythonPackages: [pythonPackages.tomlkit])}/bin/python3 ${reconcile}
    '');
  };
}
