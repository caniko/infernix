{
  description = "Evaluate Infernix Goose Home Manager wiring against a sample config";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    infernix.url = "path:../..";
    goose = {
      url = "github:caniko/goose/feat/home-manager-module";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    nixpkgs,
    home-manager,
    infernix,
    goose,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};

    sample = home-manager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        goose.homeManagerModules.default
        infernix.homeModules.default
        infernix.homeModules.goose
        {
          home.username = "tester";
          home.homeDirectory = "/home/tester";
          home.stateVersion = "24.11";

          programs.goose = {
            enable = true;
            # Skip package install — we only want to exercise the config wiring.
            package = null;
            settings.GOOSE_MODE = "smart_approve";
          };

          services.infernix.endpoints = {
            atlas-swap = {
              type = "llama-swap";
              url = "http://localhost:8013";
              models.coder = {
                name = "qwen3-coder-next";
                ctxSize = 65536;
              };
            };

            nomad-ollama = {
              type = "ollama";
              url = "http://10.10.10.20:11434";
              models.fast = {
                name = "qwen2.5-coder:14b-instruct-q6_K";
                ctxSize = 32000;
              };
            };
          };

          services.infernix.goose = {
            enable = true;
            defaultEndpoint = "atlas-swap";
            defaultModel = "coder";
          };
        }
      ];
    };
  in {
    packages.${system} = {
      config-yaml =
        sample.config.xdg.configFile."goose/config.yaml".source;
      provider-json =
        sample.config.xdg.configFile."goose/custom_providers/atlas-swap.json".source;
    };

    checks.${system}.home-manager = sample.activationPackage;
  };
}
