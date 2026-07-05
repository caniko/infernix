{
  description = "infernix — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Keep the default input remote; use `--override-input mnemo path:/...`
    # when developing against a local checkout.
    # mnemo is a private repo; SSH is required for authentication.
    # embr and visual-rubric below use HTTPS since they are public.
    mnemo = {
      url = "git+ssh://git@codeberg.org/caniko/mnemo.git";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rs-harbor.inputs.nixpkgs.follows = "nixpkgs";
    };

    # embr: code embedding indexer (replaces the old nushell indexer).
    # Lives in its own repo so it can be used standalone.
    embr = {
      url = "git+https://codeberg.org/caniko/rs-embr.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    visual-rubric = {
      url = "git+https://codeberg.org/caniko/visual-rubric.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    plinth = {
      url = "git+https://codeberg.org/caniko/plinth.git?ref=refs/heads/trunk";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rs-harbor = {
      url = "git+https://codeberg.org/caniko/rs-harbor.git?ref=trunk";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    fleetix = {
      url = "git+https://codeberg.org/caniko/fleetix.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nix-pklx = {
      url = "git+https://codeberg.org/caniko/nix-pklx.git";
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

  outputs =
    { self
    , nixpkgs
    , mnemo
    , embr
    , visual-rubric
    , plinth
    , rs-harbor
    , fleetix
    , nix-pklx
    , rust-overlay
    , hermes-agent
    , hermes-webui
    ,
    }:
    let
      # infernix's outputs serve AI/ML hosts with discrete GPUs (CUDA on
      # NVIDIA, ROCm on AMD) and llama.cpp/ollama builds whose upstreams
      # only ship x86_64 in practice. No aarch64-linux consumer exists,
      # so evaluating aarch64 outputs is dead weight that doubles
      # `nix flake check` heap for nothing.
      systems = [ "x86_64-linux" ];
      packageSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      forAllPackageSystems = nixpkgs.lib.genAttrs packageSystems;

      mkLbPackageForPkgs = pkgs: pkgs.callPackage ./packages/infernix-lb.nix { };

      mkLbPackageWithCrane = pkgs:
        let
          toolchain = rs-harbor.lib.mkToolchain { inherit pkgs; };
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
          overlays = [ rust-overlay.overlays.default ];
        });
    in
    {
      lib = {
        inherit mkLbPackageForPkgs;
      };

      nixosModules = {
        default =
          { lib
          , pkgs
          , ...
          }: {
            imports = [
              ./modules/nixos
              # Re-export upstream NixOS modules under the same default import
              # path so consumers get their options for free.
              embr.nixosModules.default
              hermes-agent.nixosModules.default
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

        mnemo = { ... }: {
          imports = [ ./modules/nixos/mnemo.nix ];
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

      packages = forAllPackageSystems (system:
        let
          infernix-lb = mkLbPackage system;
          mnemoPackage =
            if
              builtins.hasAttr "packages" mnemo
              && builtins.hasAttr system mnemo.packages
              && builtins.hasAttr "default" mnemo.packages.${system}
            then mnemo.packages.${system}.default
            else null;
          visualRubricPackage =
            if
              builtins.hasAttr "packages" visual-rubric
              && builtins.hasAttr system visual-rubric.packages
              && builtins.hasAttr "default" visual-rubric.packages.${system}
            then visual-rubric.packages.${system}.default
            else null;
          website =
            if system == "x86_64-linux"
            then
              plinth.lib.${system}.mkProjectSite
                {
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

      apps = forAllSystems (system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          deploy-pages = plinth.lib.${system}.mkDeployPagesApp {
            domain = "infernix.tartanoglu.com";
          };

          hermes-models-export = {
            type = "app";
            program = "${pkgs.writeShellApplication {
          name = "hermes-models-export";
          runtimeInputs = [
            nix-pklx.packages.${system}.pklx
            pkgs.coreutils
          ];
          text = ''
            input="''${1:-lib/hermes/ModelRouting.pkl}"
            output="''${2:-lib/hermes/model-routing.nix}"
            tmp="$(mktemp)"
            trap 'rm -f "$tmp"' EXIT
            pklx eval "$input" -o "$tmp"
            mv "$tmp" "$output"
            echo "Wrote $output from $input"
          '';
        }}/bin/hermes-models-export";
          };
        });

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixpkgs-fmt);

      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ rust-overlay.overlays.default ];
          };
          toolchain = rs-harbor.lib.mkToolchain { inherit pkgs; };
          cross = rs-harbor.lib.mkCross { inherit pkgs system; };
        in
        {
          docs = rs-harbor.lib.mkDocsShell {
            inherit pkgs cross;
            inherit (toolchain) craneLib;
            packages = [ plinth.packages.${system}.plinth-project ];
            extraShellHook = ''
              echo "Project site: plinth-project serve --config website/plinth-project.toml"
            '';
          };
        });

      checks = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
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
                  nodes = fleetix.lib.adapters.infernix.mkFleetNodes {
                    topology = {
                      hosts.atlas = {
                        network = {
                          lanIp = "192.168.178.88";
                          directLinkIp = "10.10.0.1";
                        };
                        links.wg-home.address = "10.123.0.5";
                      };
                      services.reverseProxyServices = [ ];
                    };
                    nodes.atlas = {
                      priority = 30;
                      models.qwen3-vl-8b = {
                        name = "qwen3-vl-8b";
                        capabilities = [ "chat" ];
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
                  environmentFiles = [ "/run/secrets/hermes-env" ];
                  modelRouting = {
                    enable = true;
                    profile = {
                      providers = {
                        cloud-router = {
                          urlSource = "cloudRouter";
                          models = {
                            "deepseek-v4-flash" = { };
                            "mimo-v2.5-pro" = { };
                          };
                        };
                        local-fleet = {
                          urlSource = "fleetLoadBalancer";
                          defaultModel = "qwen3-vl-8b";
                          models."qwen3-vl-8b".context_length = 4096;
                        };
                      };
                      model = {
                        default = "gpt-5.5";
                        provider = "openai-codex";
                      };
                      modelAliases."vision-local" = {
                        model = "qwen3-vl-8b";
                        provider = "local-fleet";
                      };
                      fallbackModel = [
                        {
                          model = "deepseek-v4-flash";
                          provider = "cloud-router";
                        }
                      ];
                      auxiliary = {
                        vision = {
                          model = "qwen3-vl-8b";
                          provider = "local-fleet";
                        };
                        compression = {
                          model = "deepseek-v4-flash";
                          provider = "cloud-router";
                        };
                      };
                    };
                  };
                  settings = {
                    toolsets = [ "all" ];
                  };
                };
              }
            ];
          };
          fleetLegacyLanIpSample = nixpkgs.lib.nixosSystem {
            inherit system;
            modules = [
              self.nixosModules.default
              {
                system.stateVersion = "24.11";

                services.infernix.fleet = {
                  nodes.atlas = {
                    lanIp = "192.168.178.88";
                    models.qwen3-vl-8b = {
                      name = "qwen3-vl-8b";
                      capabilities = [ "chat" ];
                    };
                  };
                  loadBalancer.enable = true;
                };
              }
            ];
          };
          pinkRavenWorkloadSample = nixpkgs.lib.nixosSystem {
            inherit system;
            modules = [
              self.nixosModules.pink-raven-workload
              ({ lib, ... }: {
                options.services.pink-raven = lib.mkOption {
                  type = lib.types.attrs;
                  default = { };
                  description = "Dummy Pink Raven option tree for workload module checks.";
                };

                config = {
                  system.stateVersion = "24.11";
                  services.infernix.workloads.pinkRaven.enable = true;
                };
              })
            ];
          };
          llamaSwapExtraFilesSample = nixpkgs.lib.nixosSystem {
            inherit system;
            modules = [
              self.nixosModules.default
              {
                system.stateVersion = "24.11";

                services.infernix.gpu = {
                  vendor = "cpu";
                  inherit pkgs;
                };

                services.infernix.llama-swap = {
                  enable = true;
                  modelsDir = "/var/lib/infernix-models";
                  models.test-model = {
                    repo = "example/main-model";
                    file = "main.gguf";
                    ctxSize = 2048;
                    extraFiles = [
                      {
                        repo = "example/main-model";
                        file = "mmproj-main.gguf";
                      }
                    ];
                  };
                };
              }
            ];
          };
          llamaSwapDownloadScript =
            pkgs.writeText
              "infernix-download-extra-files-script"
              llamaSwapExtraFilesSample.config.systemd.services.infernix-download.script;
        in
        {
          embr-wrapper = pkgs.runCommand "infernix-embr-wrapper-check" { } ''
            test "${sample.config.services.embr.qdrant.url}" = "http://127.0.0.1:6333"
            test "${sample.config.services.embr.embedding.url}" = "http://127.0.0.1:11434"
            test "${sample.config.services.embr.package}" = "${embr.packages.${system}.embr}"
            touch "$out"
          '';

          pink-raven-workload = pkgs.runCommand "infernix-pink-raven-workload-check" { } ''
            test "${pinkRavenWorkloadSample.config.services.pink-raven.embeddingBackend}" = "http"
            test "${pinkRavenWorkloadSample.config.services.pink-raven.embeddingModel}" = "qwen3-embedding-8b"
            test "${pinkRavenWorkloadSample.config.services.pink-raven.settings.PINK_RAVEN_EMBEDDING_TIMEOUT_MS}" = "180000"
            touch "$out"
          '';

          hermes-agent = pkgs.runCommand "infernix-hermes-agent-check" { } ''
            settings='${builtins.toJSON hermesAgentSample.config.services.hermes-agent.settings}'
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.model.default == "gpt-5.5"'
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.custom_providers[] | select(.name == "local-fleet" and .base_url == "http://192.168.178.31:8014/v1" and .models."qwen3-vl-8b".context_length == 4096)'
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.custom_providers[] | select(.name == "cloud-router" and .base_url == "http://127.0.0.1:2099/v1")'
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.auxiliary.vision.model == "qwen3-vl-8b"'
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.fallback_model[0].provider == "cloud-router"'
            test "${hermesAgentSample.config.services.infernix.fleet.nodes.atlas.address}" = "192.168.178.88"
            test "${hermesAgentSample.config.services.infernix.loadBalancer.backends.atlas.baseUrl}" = "http://192.168.178.88:8013"
            test "${fleetLegacyLanIpSample.config.services.infernix.loadBalancer.backends.atlas.baseUrl}" = "http://192.168.178.88:8013"
            test "${hermesAgentSample.config.services.hermes-agent.user}" = "hermes"
            test "${hermesAgentSample.config.services.hermes-agent.group}" = "hermes"
            touch "$out"
          '';

          hermes-model-routing-sidecar = pkgs.runCommand "infernix-hermes-model-routing-sidecar-check"
            {
              SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
              nativeBuildInputs = [
                nix-pklx.packages.${system}.pklx
                pkgs.cacert
                pkgs.diffutils
                pkgs.jq
                pkgs.nix
              ];
            } ''
            export NIX_STATE_DIR="$TMPDIR/nix-state"
            export NIX_LOG_DIR="$TMPDIR/nix-log"
            export NIX_CONF_DIR="$TMPDIR/nix-conf"
            mkdir -p "$NIX_STATE_DIR" "$NIX_LOG_DIR" "$NIX_CONF_DIR"

            pklx eval ${./lib/hermes/ModelRouting.pkl} -o actual.nix
            nix-instantiate --eval --json --strict actual.nix | jq -S -c . > actual.json
            nix-instantiate --eval --json --strict ${./lib/hermes/model-routing.nix} | jq -S -c . > expected.json
            diff -u actual.json expected.json
            touch "$out"
          '';

          llama-swap-extra-files = pkgs.runCommand "infernix-llama-swap-extra-files-check" { } ''
            grep -Fq 'expected_files["main.gguf"]=1' ${llamaSwapDownloadScript}
            grep -Fq 'expected_files["mmproj-main.gguf"]=1' ${llamaSwapDownloadScript}
            grep -Fq 'Downloading main.gguf from example/main-model' ${llamaSwapDownloadScript}
            grep -Fq 'Downloading mmproj-main.gguf from example/main-model' ${llamaSwapDownloadScript}
            test "${toString (builtins.length llamaSwapExtraFilesSample.config.systemd.services.infernix-download.restartTriggers)}" = "1"
            touch "$out"
          '';
        });
    };
}
