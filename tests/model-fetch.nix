# Pinned model fetch units and the on-demand ComfyUI proxy: unit wiring, plus
# the fetcher's offline skip decision (a verified manifest skips, a corrupt
# file does not).
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
    pkgs.runCommand "infernix-model-fetch-check" {HF_HUB_OFFLINE = "1";} ''
      set -eu
      job=${lib.escapeShellArg (lib.last (lib.splitString " " fetch))}
      bin=${lib.escapeShellArg (lib.head (lib.splitString " " fetch))}
      dest=$PWD/models
      mkdir -p "$dest/sub"
      printf hello > "$dest/sub/model.gguf"
      sha=$(sha256sum "$dest/sub/model.gguf" | cut -d' ' -f1)
      ${pkgs.jq}/bin/jq --arg d "$dest" '.publish.destDir = $d | .stagingDir = "\($d)/.staging-fixture" | .reserveBytes = 1' "$job" > job.json
      ${pkgs.jq}/bin/jq -n --arg sha "$sha" '{repo: "example/model", rev: "0123456789abcdef0123456789abcdef01234567", files: [{name: "sub/model.gguf", sizeBytes: 5, sha256: $sha}]}' > "$dest/ready.json"
      "$bin" job.json | grep -F "already provisioned"
      printf world > "$dest/sub/model.gguf"
      if "$bin" job.json > out.log 2>&1; then echo "corrupt file was accepted" >&2; exit 1; fi
      grep -F "hash mismatch" out.log
      touch "$out"
    ''
