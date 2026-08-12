{
  config,
  lib,
  pkgs,
  infernixOpenPencil ? null,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.openpencil;
  manifest =
    if
      infernixOpenPencil
      != null
      && infernixOpenPencil ? lib
      && infernixOpenPencil.lib ? integrationManifest
    then infernixOpenPencil.lib.integrationManifest.integration
    else null;
  systemPackages =
    if manifest != null && infernixOpenPencil.packages ? ${pkgs.stdenv.hostPlatform.system}
    then infernixOpenPencil.packages.${pkgs.stdenv.hostPlatform.system}
    else {};
  defaultPackage =
    if
      manifest
      != null
      && manifest ? packages
      && manifest.packages ? prebuiltRuntime
      && systemPackages ? ${manifest.packages.prebuiltRuntime}
    then systemPackages.${manifest.packages.prebuiltRuntime}
    else null;
  manifestHarnesses =
    if manifest != null && manifest ? harnesses
    then manifest.harnesses
    else {};
  manifestAdapter = adapter: let
    format = adapter.format or null;
    root =
      if adapter ? root
      then
        if builtins.isList adapter.root
        then adapter.root
        else [adapter.root]
      else ["mcpServers"];
    supported = lib.elem format ["json" "toml"];
  in
    if supported
    then {
      adapters.mcp = {
        inherit format root;
        configPath = adapter.configPath;
        serverKey = adapter.serverKey or null;
      };
    }
    else {
      unsupported = "OpenPencil adapter format '${toString format}' is not supported by Infernix's JSON/TOML MCP registry.";
    };
  registeredHarnesses = lib.mapAttrs (_: manifestAdapter) manifestHarnesses;
in {
  options.services.infernix.openpencil = {
    enable = mkEnableOption "OpenPencil MCP integration";
    package = mkOption {
      type = types.nullOr types.package;
      default = defaultPackage;
      description = "OpenPencil runtime package selected from the locked integration manifest.";
    };
    transport = mkOption {
      type = types.enum ["stdio" "http"];
      default = "stdio";
    };
    document = mkOption {
      type = types.str;
      default = "${config.xdg.dataHome}/openpencil/agent.op";
    };
    live.port = mkOption {
      type = types.port;
      default = 3100;
    };
    environment = mkOption {
      type = types.attrsOf types.str;
      default = {};
    };
    harnesses = mkOption {
      type = types.nullOr (types.listOf types.str);
      default = null;
      description = ''
        Optional harness allowlist. Null lets the registry select every
        detected OpenPencil adapter; use services.infernix.harnesses.<name>.mode
        = "force" for config-only clients.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions =
      [
        {
          assertion = manifest != null;
          message = "services.infernix.openpencil requires an OpenPencil flake exposing lib.integrationManifest.";
        }
        {
          assertion = cfg.package != null;
          message = "services.infernix.openpencil package is unavailable for this system; set package explicitly or disable the integration.";
        }
      ]
      ++ lib.optionals (cfg.harnesses != null) (map (name: {
          assertion = builtins.hasAttr name manifestHarnesses;
          message = "services.infernix.openpencil.harnesses contains unknown OpenPencil harness '${name}'.";
        })
        cfg.harnesses);

    services.infernix.harnesses = registeredHarnesses;
    services.infernix.mcp.servers.openpencil = {
      transport = cfg.transport;
      command = "${cfg.package}/bin/${manifest.executables.desktop}";
      args = ["--mcp" cfg.document];
      url = lib.mkIf (cfg.transport == "http") "http://127.0.0.1:${toString cfg.live.port}/mcp";
      inherit (cfg) environment harnesses;
    };

    home.activation.infernixOpenPencil = lib.hm.dag.entryAfter ["writeBoundary"] ''
      install -d -m 0755 "$(dirname ${lib.escapeShellArg cfg.document})"
      if [ ! -e ${lib.escapeShellArg cfg.document} ]; then
        install -m 0644 "${cfg.package}/${manifest.documentTemplate}" ${lib.escapeShellArg cfg.document}
      fi
    '';

    systemd.user.services.infernix-openpencil-mcp = lib.mkIf (cfg.transport == "http") {
      Unit.Description = "OpenPencil live MCP server";
      Service = {
        ExecStart = "${cfg.package}/bin/${manifest.executables.desktop} --live-mcp ${toString cfg.live.port} ${cfg.document}";
        Restart = "on-failure";
      };
      Install.WantedBy = ["default.target"];
    };
  };
}
