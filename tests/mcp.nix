{
  pkgs,
  homeManager,
  fleetix,
}: let
  inherit ((homeManager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        fleetix.homeModules.mcp
        ../modules/home-manager/harnesses.nix
        ../modules/home-manager/mcp.nix
        {
          _module.args.infernixFleetixLib = fleetix.lib;
          home = {
            username = "tester";
            homeDirectory = "/home/tester";
            stateVersion = "24.11";
          };
          services.infernix = {
            mcp.servers = {
              local.command = "/bin/example";
              remote = {
                transport = "http";
                url = "http://localhost/mcp";
              };
            };
            harnesses.codex.mode = "force";
            harnesses.claude.mode = "off";
          };
        }
      ];
    })) config;
  inherit (pkgs) lib;
in
  assert builtins.all (a: a.assertion) config.assertions;
  assert config.fleetix.mcp.enable;
  assert config.fleetix.mcp.rendered.codex.mcp_servers.local.command == "/bin/example";
  assert config.fleetix.mcp.rendered.codex.mcp_servers.remote.url == "http://localhost/mcp";
  assert !(config.fleetix.mcp.rendered ? claude);
  assert !(lib.findFirst (t: t.name == "codex") null config.fleetix.mcp.manifest.targets).autoDetect;
  assert !(config.home.activation ? infernixMcp);
  assert !(config.home.activation ? infernixHarnessRegistry); true
