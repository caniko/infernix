{
  description = "Evaluate infernix yeeHaw Home Manager wiring against a sample config";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    infernix.url = "path:../..";
    yeehaw = {
      url = "path:../../ai-yolo-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    home-manager,
    infernix,
    yeehaw,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};
    fakeYh = pkgs.writeShellScriptBin "yh" ''
      exit 0
    '';

    sample = home-manager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        yeehaw.homeManagerModules.default
        infernix.homeModules.default
        infernix.homeModules.yeehaw
        {
          home.username = "tester";
          home.homeDirectory = "/home/tester";
          home.stateVersion = "24.11";
          programs.yh.package = fakeYh;

          programs.yh = {
            enable = true;
            barns.dev = {
              session = {
                execution = "local-llama-swap-coder";
                planning = "local-llama-swap-coder";
                verification = "local-llama-swap-coder";
              };
              scout = {
                scan = "local-llama-swap-coder";
                deep = "local-llama-swap-coder";
                promote = "local-llama-swap-coder";
                triage = "local-llama-swap-coder";
              };
              ops = {
                sentinel = "local-llama-swap-coder";
                merge = "local-llama-swap-coder";
                repair = "local-llama-swap-coder";
                repairPlan = "local-llama-swap-coder";
                eval = "local-llama-swap-coder";
              };
              skill = {
                workspaceCheck = "local-llama-swap-coder";
                designToPlan = "local-llama-swap-coder";
              };
            };
          };

          services.infernix.endpoints = {
            local-ollama = {
              type = "ollama";
              url = "http://localhost:11434";
              models.fast = {
                name = "qwen2.5-coder:14b-instruct-q6_K";
                ctxSize = 32000;
              };
            };

            local-llama-swap = {
              type = "llama-swap";
              url = "http://localhost:8013";
              containerUrl = "http://host.docker.internal:8013/v1";
              models.coder = {
                name = "qwen3-coder-next";
                ctxSize = 65536;
                blockingGroup = "local-gpu";
              };
            };
          };

          services.infernix.yeehaw.enable = true;
        }
      ];
    };

    configToml = sample.config.xdg.configFile."yh/config.toml".source;
  in {
    packages.${system}.config-toml = configToml;

    checks.${system} = {
      home-manager = sample.activationPackage;
      generated-steeds = pkgs.runCommand "infernix-yeehaw-generated-steeds" {} ''
        test -e ${configToml}
        grep -Fq "local-ollama-fast" ${configToml}
        grep -Fq "local-llama-swap-coder" ${configToml}
        touch "$out"
      '';
    };
  };
}
