{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkOption types;

  mcpAdapter = types.submodule ({...}: {
    options = {
      format = mkOption {
        type = types.enum ["json" "toml"];
        description = "File format used by the harness MCP configuration.";
      };
      configPath = mkOption {
        type = types.str;
        description = "User configuration file that receives MCP registrations.";
      };
      root = mkOption {
        type = types.listOf types.str;
        default = ["mcpServers"];
        description = "Nested table path containing MCP server entries.";
      };
      serverKey = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Optional per-harness key used for MCP server registration.";
      };
    };
  });

  harness = types.submodule ({name, ...}: {
    options = {
      mode = mkOption {
        type = types.enum ["auto" "force" "off"];
        default = "auto";
        description = ''
          Harness activation policy. `auto` probes the activation PATH,
          `force` configures the harness without a command probe, and `off`
          suppresses every integration for this harness.
        '';
      };
      probe.commands = mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Executable names; any command found during activation detects the harness.";
      };
      unsupported = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Reason this harness has no supported generic file adapter.";
      };
      adapters.mcp = mkOption {
        type = types.nullOr mcpAdapter;
        default = null;
        description = "MCP file adapter contributed by an integration module.";
      };
    };
  });

  # These are the command names that are stable enough to probe centrally.
  # Harnesses without a CLI (or with a product-specific command name) remain
  # usable through mode = "force" or a consumer-provided probe.commands list.
  probeDefaults = {
    claude = ["claude"];
    codex = ["codex"];
    gemini = ["gemini"];
    hermes = ["hermes"];
    kiro = ["kiro-cli"];
    opencode = ["opencode"];
    copilot = ["copilot"];
  };

  cfg = config.services.infernix.harnesses;
  registryCfg = config.services.infernix.harnessRegistry;
  registryPlan = {
    version = 1;
    stateFile = registryCfg.statusFile;
    harnesses =
      lib.mapAttrs (_: value: {
        inherit (value) mode unsupported;
        probeCommands = value.probe.commands;
        adapters = lib.optionalAttrs (value.adapters.mcp != null) {
          mcp = value.adapters.mcp;
        };
      })
      cfg;
  };
  resolver = pkgs.writeText "infernix-harness-resolve.py" ''
    import json
    import os
    import pathlib
    import shutil
    import tempfile


    def expand_path(value):
        path = pathlib.Path(os.path.expandvars(os.path.expanduser(value)))
        return path if path.is_absolute() else pathlib.Path.home() / path


    def atomic_json(path, document):
        path = expand_path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".infernix-", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


    plan = json.loads(os.environ["INFERNIX_HARNESS_PLAN"])
    status = {
        "version": plan["version"],
        "active": [],
        "detected": [],
        "forced": [],
        "disabled": [],
        "skipped": [],
        "unsupported": [],
        "failed": [],
        "harnesses": {},
    }

    for name, definition in sorted(plan["harnesses"].items()):
        mode = definition["mode"]
        commands = definition.get("probeCommands", [])
        unsupported = definition.get("unsupported")
        detail = {"mode": mode, "active": False, "commands": commands}

        if mode == "off":
            status["disabled"].append(name)
            detail["reason"] = "disabled"
        elif unsupported:
            status["unsupported"].append(name)
            detail["reason"] = unsupported
        elif mode == "force":
            status["forced"].append(name)
            status["active"].append(name)
            detail["active"] = True
            detail["reason"] = "forced"
        else:
            matches = [command for command in commands if shutil.which(command)]
            detail["matchedCommands"] = matches
            if matches:
                status["detected"].append(name)
                status["active"].append(name)
                detail["active"] = True
                detail["reason"] = "detected"
            elif commands:
                status["skipped"].append(name)
                detail["reason"] = "not-found"
            else:
                status["unsupported"].append(name)
                detail["reason"] = "no-command-probe"

        status["harnesses"][name] = detail

    for key in ("active", "detected", "forced", "disabled", "skipped", "unsupported"):
        status[key] = sorted(status[key])
    atomic_json(os.environ["INFERNIX_HARNESS_STATUS"], status)
  '';
in {
  options.services.infernix.harnesses = mkOption {
    type = types.attrsOf harness;
    default = {};
    description = ''
      Harness registry shared by Infernix integrations. Entries are composed
      by modules; activation probes the current Home Manager profile and PATH
      instead of guessing from Nix evaluation alone.
    '';
  };

  options.services.infernix.harnessRegistry = {
    statusFile = mkOption {
      type = types.str;
      default = "${config.xdg.stateHome}/infernix/harnesses.json";
      description = "Activation status file for detected and configured harnesses.";
    };
    plan = mkOption {
      type = types.attrs;
      readOnly = true;
      description = "Declarative harness registry plan before runtime detection.";
    };
  };

  config = {
    services.infernix.harnesses =
      lib.mapAttrs (_: commands: {
        probe.commands = lib.mkDefault commands;
      })
      probeDefaults;
    services.infernix.harnessRegistry.plan = registryPlan;

    home.activation.infernixHarnessRegistry = lib.hm.dag.entryAfter ["writeBoundary"] ''
      export PATH=${lib.escapeShellArg "${config.home.profileDirectory}/bin"}:$PATH
      INFERNIX_HARNESS_PLAN=${lib.escapeShellArg (builtins.toJSON registryPlan)} \
        INFERNIX_HARNESS_STATUS=${lib.escapeShellArg registryCfg.statusFile} \
        ${pkgs.python3}/bin/python3 ${resolver}
    '';
  };
}
