# Pinned model fetch units and the on-demand ComfyUI proxy wiring, plus the
# packaged fetcher's fixtures against a stub huggingface_hub.
{
  pkgs,
  nixpkgs,
  nixosModule,
}: let
  inherit (pkgs) lib;
  inherit
    (nixpkgs.lib.nixosSystem {
      inherit (pkgs.stdenv.hostPlatform) system;
      modules = [
        nixosModule
        {
          system.stateVersion = "24.11";
          services.infernix.gpu = {
            vendor = "cpu";
            inherit pkgs;
          };
          services.infernix.comfyui = {
            enable = true;
            package = pkgs.hello;
            proxy = {
              listenStreams = ["10.0.0.5:8190"];
              onDemand = true;
            };
          };
          services.infernix.modelFetch.jobs.fixture = {
            description = "fixture";
            repo = "example/model";
            rev = "0123456789abcdef0123456789abcdef01234567";
            files = [{name = "sub/model.gguf";}];
            totalBytes = 5;
            kind = "gguf-file";
            destDir = "/var/lib/models";
          };
        }
      ];
    })
    config
    ;
  units = config.systemd.services;
  fetch = units.model-fetch-fixture.serviceConfig.ExecStart;
in
  assert lib.hasInfix "/bin/model-fetch /nix/store/" fetch;
  assert !(units.model-fetch-fixture ? wantedBy) || units.model-fetch-fixture.wantedBy == [];
  assert config.systemd.sockets.comfyui-proxy.listenStreams == ["10.0.0.5:8190"];
  assert lib.hasInfix "--exit-idle-time=20min" units.comfyui-proxy.serviceConfig.ExecStart;
  assert units.comfyui.unitConfig.StopWhenUnneeded;
  assert lib.hasInfix "/system_stats" units.comfyui.serviceConfig.ExecStartPost;
    pkgs.runCommand "infernix-model-fetch-check" {nativeBuildInputs = [pkgs.jq];} ''
      bash ${./fixtures.sh} ${lib.escapeShellArg (lib.head (lib.splitString " " fetch))} ${./.}
      touch "$out"
    ''
