{
  pkgs,
  homeManager,
  fleetix,
}: let
  mkHome = extra:
    homeManager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        fleetix.homeModules.mcp
        ../modules/home-manager/headroom.nix
        {
          home = {
            username = "tester";
            homeDirectory = "/home/tester";
            stateVersion = "24.11";
          };
          services.infernix.headroom = {
            enable = true;
            package = pkgs.hello;
            instances = {
              codex = {
                port = 8787;
                openaiUrl = "https://api.openai.com/v1";
              };
              claude = {
                port = 8788;
                anthropicUrl = "http://127.0.0.1:3456";
                requires = ["claude-code-router.service"];
              };
            };
            mcp.enable = true;
          };
        }
        extra
      ];
    };
  home = mkHome {};
  cfg = home.config;
  unit = cfg.systemd.user.services.infernix-headroom-codex;
  env = unit.Service.Environment;
  duplicate = mkHome {services.infernix.headroom.instances.claude.port = pkgs.lib.mkForce 8787;};
  duplicateValid = builtins.tryEval (assert builtins.all (a: a.assertion) duplicate.config.assertions; true);
  optedOut = mkHome {fleetix.mcp.harnesses.codex.enable = pkgs.lib.mkDefault false;};
in
  assert builtins.all (a: a.assertion) cfg.assertions;
  assert !duplicateValid.success;
  assert !optedOut.config.fleetix.mcp.harnesses.codex.enable;
  assert builtins.elem "HEADROOM_BEACON=off" env;
  assert builtins.elem "HEADROOM_CCR_BACKEND=sqlite" env;
  assert builtins.elem "HEADROOM_CCR_SQLITE_PATH=/home/tester/.local/state/infernix-headroom/ccr.sqlite" env;
  assert unit.Service.UMask == "0077";
  assert cfg.services.infernix.headroom.endpoints.codex.url == "http://127.0.0.1:8787";
  assert cfg.systemd.user.services.infernix-headroom-claude.Unit.Requires == ["claude-code-router.service"];
  assert cfg.fleetix.mcp.servers.headroom.args == ["mcp" "serve" "--proxy-url" "http://127.0.0.1:8787"]; true
