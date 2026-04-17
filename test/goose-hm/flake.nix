{
  description = "Evaluate goose home-manager module against a sample config";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    goose = {
      url = "github:caniko/goose/feat/home-manager-module";
      # Don't follow nixpkgs — goose pins its own for the package build.
    };
  };

  outputs = {
    self,
    nixpkgs,
    home-manager,
    goose,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {inherit system;};

    sample = home-manager.lib.homeManagerConfiguration {
      inherit pkgs;
      modules = [
        goose.homeManagerModules.default
        {
          home.username = "tester";
          home.homeDirectory = "/home/tester";
          home.stateVersion = "24.11";

          programs.goose = {
            enable = true;
            # Skip package install — we only want to exercise the config wiring.
            package = null;

            settings = {
              GOOSE_PROVIDER = "ollama";
              GOOSE_MODEL = "qwen3-coder";
              GOOSE_MODE = "smart_approve";

              extensions = {
                developer = {
                  enabled = true;
                  type = "builtin";
                  name = "developer";
                };

                memory = {
                  enabled = true;
                  type = "stdio";
                  name = "memory";
                  cmd = "uvx";
                  args = ["mcp-server-memory"];
                  timeout = 300;
                };
              };
            };

            customProviders = {
              llama-swap-local = {
                name = "llama-swap-local";
                engine = "openai";
                display_name = "Local llama-swap";
                base_url = "http://localhost:8013/v1";
                models = [
                  {
                    name = "qwen3-coder-next";
                    context_limit = 65536;
                  }
                ];
                supports_streaming = true;
                requires_auth = false;
              };
            };
          };
        }
      ];
    };
  in {
    # Expose the generated on-disk files so we can `nix build` and inspect them.
    packages.${system} = {
      config-yaml =
        sample.config.xdg.configFile."goose/config.yaml".source;
      provider-json =
        sample.config.xdg.configFile."goose/custom_providers/llama-swap-local.json".source;
    };

    # Full HM activation package — proves the whole module evaluates cleanly.
    checks.${system}.home-manager = sample.activationPackage;
  };
}
