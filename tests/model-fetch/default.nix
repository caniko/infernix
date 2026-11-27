# Pinned model fetch units and the on-demand ComfyUI proxy wiring, plus the
# packaged fetcher's fixtures against a stub huggingface_hub.
{
  pkgs,
  nixpkgs,
  nixosModule,
}: let
  inherit (pkgs) lib;
  modelFetch = pkgs.callPackage ../../packages/model-fetch {};
  lockPath = "/var/lib/shared-models.doty-lock";
  anchor = "f ${lockPath} 0644 root root - -";
  job = {
    description = "fixture";
    repo = "example/model";
    rev = "0123456789abcdef0123456789abcdef01234567";
    files = [{name = "sub/model.gguf";}];
    totalBytes = 5;
    kind = "gguf-file";
    destDir = "/var/lib/models";
  };
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
          services.infernix.modelFetch.jobs = {
            fixture = job;
            shared = job // {inherit lockPath;};
            shared-other = job // {inherit lockPath;};
          };
        }
      ];
    })
    config
    ;
  units = config.systemd.services;
  fetch = units.model-fetch-fixture.serviceConfig.ExecStart;
  fetchCommand = "${modelFetch}/bin/model-fetch";
in
  assert lib.hasInfix "-model-lock.py --exclusive /var/lib/models.doty-lock -- /nix/store/" fetch;
  assert lib.hasInfix "-model-lock.py --exclusive ${lockPath} -- /nix/store/" units.model-fetch-shared.serviceConfig.ExecStart;
  assert builtins.elem "f /var/lib/models.doty-lock 0644 root root - -" config.systemd.tmpfiles.rules;
  assert builtins.filter (rule: rule == anchor) config.systemd.tmpfiles.rules == [anchor];
  assert lib.hasInfix "/bin/model-fetch /nix/store/" fetch;
  assert !(units.model-fetch-fixture ? wantedBy) || units.model-fetch-fixture.wantedBy == [];
  assert config.systemd.sockets.comfyui-proxy.listenStreams == ["10.0.0.5:8190"];
  assert lib.hasInfix "--exit-idle-time=20min" units.comfyui-proxy.serviceConfig.ExecStart;
  assert units.comfyui.unitConfig.StopWhenUnneeded;
  assert lib.hasInfix "/system_stats" units.comfyui.serviceConfig.ExecStartPost;
    pkgs.runCommand "infernix-model-fetch-check" {nativeBuildInputs = [pkgs.jq];} ''
      bash ${./fixtures.sh} ${lib.escapeShellArg fetchCommand} ${./.}
      touch "$out"
    ''
