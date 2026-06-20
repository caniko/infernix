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

    visual-rubric = {
      url = "git+ssh://git@codeberg.org/caniko/visual-rubric.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    plinth = {
      url = "git+https://codeberg.org/caniko/plinth.git?ref=refs/heads/trunk";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rs-harbor = {
      url = "git+https://codeberg.org/caniko/rs-harbor.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rust-overlay.follows = "rs-harbor/rust-overlay";

    hermes-agent = {
      url = "github:NousResearch/hermes-agent";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    hermes-webui = {
      url = "github:caniko/hermes-webui";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.hermes-agent.follows = "hermes-agent";
    };
  };

  outputs = {
    self,
    nixpkgs,
    mnemo,
    embr,
    visual-rubric,
    plinth,
    rs-harbor,
    rust-overlay,
    hermes-agent,
    hermes-webui,
  }: let
    # infernix's outputs serve AI/ML hosts with discrete GPUs (CUDA on
    # NVIDIA, ROCm on AMD) and llama.cpp/ollama builds whose upstreams
    # only ship x86_64 in practice. No aarch64-linux consumer exists,
    # so evaluating aarch64 outputs is dead weight that doubles
    # `nix flake check` heap for nothing.
    systems = ["x86_64-linux"];
    packageSystems = [
      "x86_64-linux"
      "aarch64-linux"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
    forAllPackageSystems = nixpkgs.lib.genAttrs packageSystems;

    mkLbPackageForPkgs = pkgs: pkgs.callPackage ./packages/infernix-lb.nix {};

    mkLbPackageWithCrane = pkgs: let
      toolchain = rs-harbor.lib.mkToolchain {inherit pkgs;};
      inherit (toolchain) craneLib;
      src = craneLib.cleanCargoSource (builtins.path {
        path = ./.;
        name = "infernix-source";
      });
      commonArgs = {
        inherit src;
        pname = "infernix-lb";
        version = "0.1.0";
        strictDeps = true;
        cargoExtraArgs = "-p infernix-lb";
      };
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;
    in
      craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
        });

    mkLbPackage = system:
      mkLbPackageWithCrane (import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
      });
  in {
    lib = {
      inherit mkLbPackageForPkgs;
    };

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
          # Re-export hermes-agent's NixOS module under the same default
          # import so consumers get `services.hermes-agent.*` for free.
          hermes-agent.nixosModules.default
          # Re-export hermes-webui's NixOS module under the same default
          # import so consumers get `services.hermes-webui.*` for free.
          hermes-webui.nixosModules.default
        ];
        # Thread the locked nixos-unstable nixpkgs flake into the module tree
        # so ollama / llama-cpp / llama-swap can re-instantiate it with the
        # consumer's own system + config (GPU flags, allowUnfree, etc.).
        _module.args.infernixBleedingNixpkgs = nixpkgs;
        _module.args.infernixHermesAgent = hermes-agent;
        _module.args.infernixHermesWebui = hermes-webui;
        _module.args.infernixMnemo = mnemo;
        _module.args.infernixEmbr = embr;
        _module.args.infernixVisualRubric = visual-rubric;
        _module.args.infernixSelf = self;
        _module.args.infernixMkLbPackageForPkgs = mkLbPackageForPkgs;
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

      pink-raven-workload = ./modules/nixos/pink-raven-workload.nix;
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
      # Opt-in sub-module that writes programs.visual-rubric.* from the
      # services.infernix.visual-rubric outputs.
      visualRubric = import ./modules/home-manager/visual-rubric-programs.nix;
      # Opt-in sub-module that wires hermes CLI providers from
      # services.infernix.endpoints. Only import for users that also
      # configure services.infernix.hermes-agent.
      hermes-agent = import ./modules/home-manager/hermes-agent-programs.nix;
    };

    packages = forAllPackageSystems (system: let
      infernix-lb = mkLbPackage system;
      mnemoPackage =
        if builtins.hasAttr "packages" mnemo
        && builtins.hasAttr system mnemo.packages
        && builtins.hasAttr "default" mnemo.packages.${system}
        then mnemo.packages.${system}.default
        else null;
      visualRubricPackage =
        if builtins.hasAttr "packages" visual-rubric
        && builtins.hasAttr system visual-rubric.packages
        && builtins.hasAttr "default" visual-rubric.packages.${system}
        then visual-rubric.packages.${system}.default
        else null;
      website =
        if system == "x86_64-linux"
        then
          plinth.lib.${system}.mkProjectSite {
            pname = "infernix-website";
            domain = "infernix.tartanoglu.com";
            configPath = ./website/plinth-project.toml;
          }
        else null;
    in
      {
        inherit infernix-lb;
        default =
          if system == "x86_64-linux"
          then embr.packages.${system}.embr
          else infernix-lb;
      }
      // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
        embr = embr.packages.${system}.embr;
        website = website;
        site = website;
      }
      // nixpkgs.lib.optionalAttrs (system == "x86_64-linux" && mnemoPackage != null) {
        mnemo = mnemoPackage;
      }
      // nixpkgs.lib.optionalAttrs (system == "x86_64-linux" && visualRubricPackage != null) {
        visual-rubric = visualRubricPackage;
      });

    apps = forAllSystems (system: {
      deploy-pages = plinth.lib.${system}.mkDeployPagesApp {
        domain = "infernix.tartanoglu.com";
      };
    });

    devShells = forAllSystems (system: let
      pkgs = import nixpkgs {
        inherit system;
        overlays = [rust-overlay.overlays.default];
      };
      toolchain = rs-harbor.lib.mkToolchain {inherit pkgs;};
      cross = rs-harbor.lib.mkCross {inherit pkgs system;};
    in {
      docs = rs-harbor.lib.mkDocsShell {
        inherit pkgs cross;
        inherit (toolchain) craneLib;
        packages = [plinth.packages.${system}.plinth-project];
        extraShellHook = ''
          echo "Project site: plinth-project serve --config website/plinth-project.toml"
        '';
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
      hermesAgentSample = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.default
          {
            system.stateVersion = "24.11";

            services.infernix.fleet = {
              nodes = {
                atlas = {
                  lanIp = "192.168.178.88";
                  priority = 30;
                  models.qwen3-vl-8b = {
                    name = "qwen3-vl-8b";
                    capabilities = ["chat"];
                  };
                };
              };
              loadBalancer = {
                enable = true;
                host = "192.168.178.31";
                port = 8014;
              };
            };

            services.infernix.hermes-agent = {
              enable = true;
              useFleetModels = true;
              environmentFiles = ["/run/secrets/hermes-env"];
              settings = {
                model.default = "qwen3-vl-8b";
                toolsets = ["all"];
              };
            };
          }
        ];
      };
      pinkRavenWorkloadSample = nixpkgs.lib.nixosSystem {
        inherit system;
        modules = [
          self.nixosModules.pink-raven-workload
          ({lib, ...}: {
            options.services.pink-raven = lib.mkOption {
              type = lib.types.attrs;
              default = {};
              description = "Dummy Pink Raven option tree for workload module checks.";
            };

            config = {
              system.stateVersion = "24.11";
              services.infernix.workloads.pinkRaven.enable = true;
            };
          })
        ];
      };
    in {
      embr-wrapper = pkgs.runCommand "infernix-embr-wrapper-check" {} ''
        test "${sample.config.services.embr.qdrant.url}" = "http://127.0.0.1:6333"
        test "${sample.config.services.embr.embedding.url}" = "http://127.0.0.1:11434"
        test "${sample.config.services.embr.package}" = "${embr.packages.${system}.embr}"
        touch "$out"
      '';

      pink-raven-workload = pkgs.runCommand "infernix-pink-raven-workload-check" {} ''
        test "${pinkRavenWorkloadSample.config.services.pink-raven.embeddingBackend}" = "http"
        test "${pinkRavenWorkloadSample.config.services.pink-raven.embeddingModel}" = "qwen3-embedding-8b"
        test "${pinkRavenWorkloadSample.config.services.pink-raven.settings.PINK_RAVEN_EMBEDDING_TIMEOUT_MS}" = "180000"
        touch "$out"
      '';

      hermes-agent = pkgs.runCommand "infernix-hermes-agent-check" {} ''
        test "${hermesAgentSample.config.services.hermes-agent.settings.model.default}" = "qwen3-vl-8b"
        test "${hermesAgentSample.config.services.hermes-agent.settings.model.base_url}" = "http://192.168.178.31:8014/v1"
        test "${hermesAgentSample.config.services.hermes-agent.user}" = "hermes"
        test "${hermesAgentSample.config.services.hermes-agent.group}" = "hermes"
        touch "$out"
      '';
    });
  };
}
