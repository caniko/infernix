{
  description = "infernix — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Keep the default input remote; use `--override-input mnemo path:/...`
    # when developing against a local checkout.
    mnemo = {
      url = "git+ssh://git@codeberg.org/caniko/mnemo.git";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rs-harbor.inputs.nixpkgs.follows = "nixpkgs";
    };

    # embr: code embedding indexer (replaces the old nushell indexer).
    # Lives in its own repo so it can be used standalone.
    embr = {
      url = "git+ssh://git@codeberg.org/caniko/rs-embr.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    plinth = {
      url = "git+https://codeberg.org/caniko/plinth.git?ref=refs/heads/trunk";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    mnemo,
    embr,
    plinth,
  }: let
    # infernix's outputs serve AI/ML hosts with discrete GPUs (CUDA on
    # NVIDIA, ROCm on AMD) and llama.cpp/ollama builds whose upstreams
    # only ship x86_64 in practice. No aarch64-linux consumer exists,
    # so evaluating aarch64 outputs is dead weight that doubles
    # `nix flake check` heap for nothing.
    systems = ["x86_64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    nixosModules = {
      default = {
        lib,
        pkgs,
        ...
      }: {
        imports = [
          ./modules/nixos
          # Re-export embr's NixOS module under the same default import
          # path so consumers get `services.embr.*` for free.
          embr.nixosModules.default
        ];
        # Thread the locked nixos-unstable nixpkgs flake into the module tree
        # so ollama / llama-cpp / llama-swap can re-instantiate it with the
        # consumer's own system + config (GPU flags, allowUnfree, etc.).
        _module.args.infernixBleedingNixpkgs = nixpkgs;
        _module.args.infernixMnemo = mnemo;
        _module.args.infernixEmbr = embr;
        # Default `services.embr.package` to the one locked by infernix,
        # picking the binary for the active host system. mkDefault keeps
        # it overridable downstream.
        services.embr.package =
          lib.mkDefault embr.packages.${pkgs.stdenv.hostPlatform.system}.embr;
      };

      mnemo = {
        ...
      }: {
        imports = [./modules/nixos/mnemo.nix];
        _module.args.infernixMnemo = mnemo;
      };
    };

    homeModules = {
      default = import ./modules/home-manager;
      # Opt-in sub-module that writes programs.goose.* from the
      # services.infernix.goose outputs. Only import for users that also
      # import goose-hm's HM module.
      goose = import ./modules/home-manager/goose-programs.nix;
      # Opt-in sub-module that writes programs.yh.steeds from the
      # services.infernix.yeehaw outputs. Only import for users that also
      # import yeeHaw's HM module.
      yeehaw = import ./modules/home-manager/yeehaw-programs.nix;
    };

    packages = forAllSystems (system: let
      mnemoPackage =
        if builtins.hasAttr "packages" mnemo
        && builtins.hasAttr system mnemo.packages
        && builtins.hasAttr "default" mnemo.packages.${system}
        then mnemo.packages.${system}.default
        else null;
      website = plinth.lib.${system}.mkProjectSite {
        pname = "infernix-website";
        domain = "infernix.tartanoglu.com";
        configPath = ./website/plinth-project.toml;
      };
    in
      {
        embr = embr.packages.${system}.embr;
        default = embr.packages.${system}.embr;
        website = website;
        site = website;
      }
      // nixpkgs.lib.optionalAttrs (mnemoPackage != null) {
        mnemo = mnemoPackage;
      });

    apps = forAllSystems (system: {
      deploy-pages = plinth.lib.${system}.mkDeployPagesApp {
        domain = "infernix.tartanoglu.com";
      };
    });

    checks = forAllSystems (system: let
      pkgs = import nixpkgs {inherit system;};
      sample = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          {
            system.stateVersion = "24.11";

            services.infernix.gpu = {
              vendor = "cpu";
              inherit pkgs;
            };

            services.infernix.qdrant.enable = true;
            services.infernix.ollama.enable = true;

            services.infernix.embr = {
              enable = true;
              qdrant.useInfernixService = true;
              embedding.useInfernixService = true;
              projectsRoot = "/srv/projects";
              embedding.vectors = [
                {
                  name = "code";
                  model = "qwen3-embedding:8b";
                  dim = 4096;
                }
              ];
            };
          }
        ];
      };
    in {
      embr-wrapper = pkgs.runCommand "infernix-embr-wrapper-check" {} ''
        test "${sample.config.services.embr.qdrant.url}" = "http://127.0.0.1:6333"
        test "${sample.config.services.embr.embedding.url}" = "http://127.0.0.1:11434"
        test "${sample.config.services.embr.package}" = "${embr.packages.${system}.embr}"
        touch "$out"
      '';
    });
  };
}
