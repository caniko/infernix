{
  description = "infernix — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # HM modules are checked with a real Home Manager module graph. Keeping
    # this input direct prevents fixtures from accidentally omitting HM's
    # activation options and lib.hm DAG helpers.
    home-manager = {
      url = "github:nix-community/home-manager";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    visual-rubric = {
      url = "git+https://github.com/caniko/visual-rubric.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    plinth = {
      url = "git+https://github.com/caniko/plinth.git?ref=refs/heads/trunk";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    rs-harbor = {
      url = "git+https://github.com/caniko/rs-harbor.git?ref=trunk&rev=77d0a937c760e6ced8b7ec8fc5a214f550abe35e";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    fleetix = {
      url = "git+https://github.com/caniko/fleetix.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    nix-pklx = {
      url = "git+https://github.com/caniko/nix-pklx.git";
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

    # Graphify is exposed through Infernix so every supported agent harness
    # receives the same registration and package revision.
    graphify = {
      url = "github:caniko/graphify/0b1e9723577b35b974eb36441eec47a624ef0082";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self
    , nixpkgs
    , home-manager
    , visual-rubric
    , plinth
    , rs-harbor
    , fleetix
    , nix-pklx
    , rust-overlay
    , hermes-agent
    , hermes-webui
    , graphify
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

      mkCargoPackageWithCrane = { pkgs, packageName }:
        let
          toolchain = rs-harbor.lib.mkToolchain { inherit pkgs; };
          inherit (toolchain) craneLib;
          source = ./.;
          src = craneLib.cleanCargoSource source;
          commonArgs = {
            inherit src;
            rsHarborCargoTomlContents = builtins.readFile ./Cargo.toml;
            pname = packageName;
            version = "0.1.0";
            strictDeps = true;
            cargoExtraArgs = "-p ${packageName}";
          };
          cargoArtifacts = craneLib.buildDepsOnly commonArgs;
        in
        craneLib.buildPackage (commonArgs
          // {
          inherit cargoArtifacts;
        });

      mkLbPackageWithCrane = pkgs:
        mkCargoPackageWithCrane {
          inherit pkgs;
          packageName = "infernix-lb";
        };

      mkWorkerdPackageWithCrane = pkgs:
        mkCargoPackageWithCrane {
          inherit pkgs;
          packageName = "infernix-workerd";
        };

      mkLbPackage = system:
        mkLbPackageWithCrane (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        });

      mkWorkerdPackage = system:
        mkWorkerdPackageWithCrane (import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        });
    in
    {
      lib = {
        inherit mkLbPackageForPkgs;
        modelCatalog = import ./lib/model-catalog.nix { lib = nixpkgs.lib; };
      };

      nixosModules = {
        visual-rubric = {
          imports = [./modules/nixos/visual-rubric.nix];
          _module.args.infernixVisualRubric = visual-rubric;
        };
        default =
          { lib
          , pkgs
          , ...
          }: {
            imports = [
              ./modules/nixos
              # Re-export upstream NixOS modules under the same default import
              # path so consumers get their options for free.
              hermes-agent.nixosModules.default
              hermes-webui.nixosModules.default
            ]
            ++ lib.optional
              (graphify ? nixosModules && graphify.nixosModules ? default)
              graphify.nixosModules.default;
            # Thread the locked nixos-unstable nixpkgs flake into the module tree
            # so ollama / llama-cpp / llama-swap can re-instantiate it with the
            # consumer's own system + config (GPU flags, allowUnfree, etc.).
            _module.args.infernixBleedingNixpkgs = nixpkgs;
            _module.args.infernixHermesAgent = hermes-agent;
            _module.args.infernixHermesWebui = hermes-webui;
            _module.args.infernixVisualRubric = visual-rubric;
            _module.args.infernixGraphify = graphify;
            _module.args.infernixCodexAcp = self.packages.${pkgs.system}.codex-acp;
            _module.args.infernixSelf = self;
            _module.args.infernixMkLbPackageForPkgs = mkLbPackageForPkgs;
          };

        pink-raven-workload = ./modules/nixos/pink-raven-workload.nix;
      }
      // nixpkgs.lib.optionalAttrs
        (graphify ? nixosModules && graphify.nixosModules ? default)
        {
          graphify = graphify.nixosModules.default;
        };

      homeModules = {
        default = { pkgs, ... }: {
          imports = [ (import ./modules/home-manager) ];
          _module.args.infernixVisualRubric = visual-rubric;
          _module.args.infernixGraphify = graphify;
          _module.args.infernixCodexAcp = self.packages.${pkgs.system}.codex-acp;
        };
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
          pkgs = nixpkgs.legacyPackages.${system};
          infernix-lb = mkLbPackage system;
          infernix-workerd = mkWorkerdPackage system;
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
          inherit infernix-lb infernix-workerd;
          codex-acp = pkgs.callPackage ./packages/codex-acp.nix { };
          graphify =
            graphify.packages.${system}.full
              or graphify.packages.${system}.default;
          default = infernix-lb;
        }
        // nixpkgs.lib.optionalAttrs (system == "x86_64-linux") {
          website = website;
          site = website;
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
              }
            ];
          };
          graphifyNixosModuleAvailable =
            graphify ? nixosModules
            && graphify.nixosModules ? default;
          graphifyNixosSample =
            if graphifyNixosModuleAvailable
            then
              nixpkgs.lib.nixosSystem {
                inherit system;
                modules = [
                  self.nixosModules.default
                  {
                    system.stateVersion = "24.11";
                    services.graphify = {
                      enable = true;
                      instances.postgresql = {
                        source.postgresql = {
                          enable = true;
                          database = "infernix";
                        };
                        extraction.onCalendar = "daily";
                        server.enable = true;
                      };
                    };
                  }
                ];
              }
            else null;
          workloadFabricSample = nixpkgs.lib.nixosSystem {
            inherit system;
            modules = [
              self.nixosModules.default
              {
                system.stateVersion = "24.11";

                services.infernix.workloadFabric = {
                  enable = true;
                  databaseUrl = "postgres:///canix?host=/run/postgresql";
                  workerId = "atlas";
                  capabilities = [ "cpu" "semantic" ];
                  adapters.graphify = {
                    workload = "graphify";
                    queues = [ "code" "semantic" ];
                    command = "/bin/canix";
                    args = [ "graphify" "run-job" ];
                  };
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
                    moa = {
                      default_preset = "gpt55_dsflash";
                      presets = {
                        gpt55_dsflash = {
                          reference_models = [
                            {
                              model = "deepseek-v4-flash";
                              provider = "cloud-router";
                            }
                          ];
                          aggregator = {
                            model = "gpt-5.5";
                            provider = "openai-codex";
                          };
                          enabled = true;
                        };
                      };
                    };
                  };
                  scheduledSettings = {
                    enable = true;
                    timeZone = "America/Los_Angeles";
                    restartService = true;
                    profiles = {
                      day.settingsOverlay.moa.default_preset = "gpt55_mimo";
                      night.settingsOverlay.moa.default_preset = "gpt55_dsflash";
                    };
                    switches = {
                      day = {
                        profile = "day";
                        onCalendar = "*-*-* 09:00:00";
                      };
                      night = {
                        profile = "night";
                        onCalendar = "*-*-* 17:00:00";
                      };
                    };
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
          visualRubricDirectSample = home-manager.lib.homeManagerConfiguration {
            inherit pkgs;
            extraSpecialArgs = { osConfig = null; };
            modules = [
              self.homeModules.default
              {
                home = {
                  username = "tester";
                  homeDirectory = "/home/tester";
                  stateVersion = "24.11";
                };
                services.infernix.visual-rubric.enable = true;
              }
            ];
          };
          visualRubricPipelineSample = home-manager.lib.homeManagerConfiguration {
            inherit pkgs;
            extraSpecialArgs = { osConfig = null; };
            modules = [
              self.homeModules.default
              {
                home = {
                  username = "tester";
                  homeDirectory = "/home/tester";
                  stateVersion = "24.11";
                };
                services.infernix.endpoints.local-lb = {
                  type = "llama-swap";
                  url = "http://127.0.0.1:8013";
                  models.vlm = {
                    name = "qwen3-vl-8b";
                    ctxSize = 4096;
                  };
                };
                services.infernix.visual-rubric = {
                  enable = true;
                  mode = "pipeline";
                };
              }
            ];
          };
          visualRubricDirectConfig =
            visualRubricDirectSample.config.xdg.configFile."visual-rubric/config.toml".source;
          visualRubricPipelineConfig =
            visualRubricPipelineSample.config.xdg.configFile."visual-rubric/config.toml".source;
          visualRubricDirectPackage =
            pkgs.lib.findFirst
              (package: package == visual-rubric.packages.${system}."codex-acp")
              null
              visualRubricDirectSample.config.home.packages;
          visualRubricPipelinePackage =
            pkgs.lib.findFirst
              (package: package == visual-rubric.packages.${system}.default)
              null
              visualRubricPipelineSample.config.home.packages;
          graphifyHarnesses = [
            "agents"
            "aider"
            "amp"
            "antigravity"
            "claude"
            "claw"
            "codebuddy"
            "codex"
            "copilot"
            "cursor"
            "devin"
            "droid"
            "gemini"
            "hermes"
            "kilo"
            "kiro"
            "kimi"
            "opencode"
            "pi"
            "trae"
            "trae-cn"
            "vscode"
          ];
          graphifySample = home-manager.lib.homeManagerConfiguration {
            inherit pkgs;
            modules = [
              self.homeModules.default
              {
                home = {
                  username = "tester";
                  homeDirectory = "/home/tester";
                  stateVersion = "24.11";
                };
                services.infernix.endpoints.local = {
                  type = "llama-swap";
                  url = "http://127.0.0.1:8013";
                  models.dsv4.name = "dsv4";
                };
                services.infernix.graphify = {
                  enable = true;
                  endpoint = "local";
                };
              }
            ];
          };
          graphifyExpectedPackageName =
            if graphify.packages.${system} ? full
            then graphify.packages.${system}.full.name
            else "graphify-with-openai";
          graphifyAcpSample = home-manager.lib.homeManagerConfiguration {
            inherit pkgs;
            modules = [
              self.homeModules.default
              {
                home = {
                  username = "tester";
                  homeDirectory = "/home/tester";
                  stateVersion = "24.11";
                };
                services.infernix.graphify = {
                  enable = true;
                  semanticBackend = "acp";
                };
              }
            ];
          };
          graphifyRegistrationScript = pkgs.writeShellScript "infernix-graphify-harness-registration" ''
            set -eu
            export HOME="$TMPDIR/graphify-home"
            mkdir -p "$HOME"
            ${graphifySample.config.home.activation.infernixGraphify.data}
          '';
          modelCatalogSample = {
            models = {
              qwen3-vl-8b = {
                host = "atlas";
                repo = "Qwen/Qwen3-VL-8B-Instruct-GGUF";
                file = "Qwen3VL-8B-Instruct-Q8_0.gguf";
                ctxSize = 4096;
                ttl = 300;
                aliases = [ "qwen3-vl" "vlm" ];
                capabilities = [ "chat" ];
                extraFiles = [
                  {
                    repo = "Qwen/Qwen3-VL-8B-Instruct-GGUF";
                    file = "mmproj.gguf";
                  }
                ];
                extraArgs = [
                  "--mmproj {modelsDir}/mmproj.gguf"
                  "--jinja"
                ];
              };
            };
            homeManager.endpoints.atlas-lb.models.vlm.model = "qwen3-vl-8b";
            probes.atlas.chat = [
              {
                name = "qwen3-vl-8b";
                model = "qwen3-vl-8b";
                maxTokens = 10;
              }
            ];
          };
          modelCatalogLib = self.lib.modelCatalog;
          renderedLlamaSwapModels = modelCatalogLib.mkLlamaSwapModels {
            catalog = modelCatalogSample;
            host = "atlas";
            modelsDir = "/models";
          };
          renderedFleetModels = modelCatalogLib.mkFleetModels {
            catalog = modelCatalogSample;
            host = "atlas";
          };
          renderedHmModels = modelCatalogLib.mkHmEndpointModels {
            catalog = modelCatalogSample;
            endpoint = "atlas-lb";
          };
        in
        {
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
            printf '%s' "$settings" | ${pkgs.jq}/bin/jq -e '.moa.default_preset == "gpt55_dsflash"'
            test "${hermesAgentSample.config.systemd.timers."hermes-agent-scheduled-settings-day".timerConfig.OnCalendar}" = "*-*-* 09:00:00 America/Los_Angeles"
            test "${hermesAgentSample.config.systemd.timers."hermes-agent-scheduled-settings-night".timerConfig.OnCalendar}" = "*-*-* 17:00:00 America/Los_Angeles"
            case ${pkgs.lib.escapeShellArg (toString hermesAgentSample.config.systemd.services."hermes-agent-scheduled-settings-day".serviceConfig.ExecStart)} in
              *"--restart"*) ;;
              *) echo "day schedule service does not restart hermes-agent" >&2; exit 1 ;;
            esac
            test "${hermesAgentSample.config.services.infernix.fleet.nodes.atlas.address}" = "192.168.178.88"
            test "${hermesAgentSample.config.services.infernix.loadBalancer.backends.atlas.baseUrl}" = "http://192.168.178.88:8013"
            test "${fleetLegacyLanIpSample.config.services.infernix.loadBalancer.backends.atlas.baseUrl}" = "http://192.168.178.88:8013"
            test "${hermesAgentSample.config.services.hermes-agent.user}" = "hermes"
            test "${hermesAgentSample.config.services.hermes-agent.group}" = "hermes"
            touch "$out"
          '';

          visual-rubric-home = pkgs.runCommand "infernix-visual-rubric-home-check" { } ''
            grep -Fq 'mode = "direct"' ${visualRubricDirectConfig}
            grep -Fq 'backend = "${visualRubricDirectSample.config.services.infernix.acp.resolvedProviders.codex.command}"' ${visualRubricDirectConfig}
            grep -Fq 'model = "gpt-5.5"' ${visualRubricDirectConfig}
            grep -Fq 'effort = "medium"' ${visualRubricDirectConfig}
            ! grep -Fq '[vision]' ${visualRubricDirectConfig}
            test "${visualRubricDirectPackage}" = "${visual-rubric.packages.${system}."codex-acp"}"

            grep -Fq 'mode = "pipeline"' ${visualRubricPipelineConfig}
            grep -Fq 'backend = "opencode"' ${visualRubricPipelineConfig}
            grep -Fq 'args = [' ${visualRubricPipelineConfig}
            grep -Fq 'url = "http://127.0.0.1:8013"' ${visualRubricPipelineConfig}
            grep -Fq 'model = "qwen3-vl-8b"' ${visualRubricPipelineConfig}
            test "${visualRubricPipelinePackage}" = "${visual-rubric.packages.${system}.default}"
            touch "$out"
          '';

          graphify-harness-registration = pkgs.runCommand "infernix-graphify-harness-registration-check" { } ''
            ${graphifyRegistrationScript}
            ${graphifyRegistrationScript}
            expected='${builtins.toJSON graphifyHarnesses}'
            actual='${builtins.toJSON graphifySample.config.services.infernix.graphify.registeredHarnesses}'
            test "$actual" = "$expected"
            test "${graphifySample.config.services.infernix.graphify.generatedSettings.OPENAI_BASE_URL}" = "http://127.0.0.1:8013/v1"
            test "${graphifySample.config.services.infernix.graphify.generatedSettings.OPENAI_MODEL}" = "dsv4"
            test "${graphifySample.config.services.infernix.graphify.package.name}" = "${graphifyExpectedPackageName}"
            commands='${builtins.toJSON graphifySample.config.services.infernix.graphify.registrationCommands}'
            printf '%s' "$commands" | ${pkgs.jq}/bin/jq -e 'length == 22'
            printf '%s' "$commands" | ${pkgs.jq}/bin/jq -e 'all(.[]; contains("graphify"))'
            test -f "$TMPDIR/graphify-home/AGENTS.md"
            test -f "$TMPDIR/graphify-home/CLAUDE.md"
            test -f "$TMPDIR/graphify-home/.claude/settings.json"
            test -f "$TMPDIR/graphify-home/.codex/hooks.json"
            test -f "$TMPDIR/graphify-home/.gemini/settings.json"
            test -f "$TMPDIR/graphify-home/.cursor/rules/graphify.mdc"
            test -f "$TMPDIR/graphify-home/.kilo/kilo.json"
            test -f "$TMPDIR/graphify-home/.opencode/opencode.json"
            ${pkgs.jq}/bin/jq -e '.plugin | index("./plugins/graphify.js") != null' "$TMPDIR/graphify-home/.opencode/opencode.json"
            ${pkgs.jq}/bin/jq -e '.plugin | index("plugins/graphify.js") == null' "$TMPDIR/graphify-home/.opencode/opencode.json"
            ${pkgs.jq}/bin/jq -e '.plugin | index(".opencode/plugins/graphify.js") == null' "$TMPDIR/graphify-home/.opencode/opencode.json"
            test -f "$TMPDIR/graphify-home/.github/copilot-instructions.md"
            touch "$out"
          '';

          graphify-acp-provider = pkgs.runCommand "infernix-graphify-acp-provider-check" { } ''
            provider='${builtins.toJSON graphifyAcpSample.config.services.infernix.acp.resolvedProviders.codex}'
            printf '%s' "$provider" | ${pkgs.jq}/bin/jq -e '.capabilities == {"image":true,"sessionConfig":true,"text":true}'
            printf '%s' "$provider" | ${pkgs.jq}/bin/jq -e '.configOptions == {}'
            printf '%s' "$provider" | ${pkgs.jq}/bin/jq -e '.environment.CODEX_HOME == "/home/tester/.codex"'
            test "${graphifyAcpSample.config.services.infernix.graphify.generatedSettings.GRAPHIFY_SEMANTIC_BACKEND}" = acp
            test "${graphifyAcpSample.config.services.infernix.graphify.generatedSettings.GRAPHIFY_ACP_BIN}" = "${graphifyAcpSample.config.services.infernix.acp.resolvedProviders.codex.command}"
            test "${graphifyAcpSample.config.services.infernix.graphify.generatedSettings.GRAPHIFY_ACP_MODEL}" = gpt-5.5
            test '${graphifyAcpSample.config.services.infernix.graphify.generatedSettings.GRAPHIFY_ACP_CONFIG_JSON}' = '{"mode":"read-only"}'
            test "${graphifyAcpSample.config.services.infernix.graphify.package}" = "${graphify.packages.${system}.acp}"
            touch "$out"
          '';

          codex-acp-closure = let
            closure = pkgs.closureInfo {
              rootPaths = [ self.packages.${system}.codex-acp ];
            };
          in pkgs.runCommand "infernix-codex-acp-closure-check" { } ''
            package=${self.packages.${system}.codex-acp}
            test -x "$package/bin/codex-acp"
            test -f "$package/libexec/codex-acp/index.js"
            test ! -e "$package/lib/node_modules"
            test "$(find "$package" -type f | wc -l)" -eq 2
            test "$(grep -Fxc '${pkgs.codex}' ${closure}/store-paths)" -eq 1
            touch "$out"
          '';

          graphify-nixos-module =
            if graphifyNixosModuleAvailable
            then
              pkgs.runCommand "infernix-graphify-nixos-module-check" { } ''
                test "${graphifyNixosSample.config.services.graphify.instances.postgresql.source.postgresql.database}" = infernix
                test "${graphifyNixosSample.config.services.graphify.package}" = "${graphify.packages.${system}.full}"
                test "${graphifyNixosSample.config.systemd.services.graphify-postgresql.serviceConfig.User}" = graphify
                test "${toString graphifyNixosSample.config.systemd.services.graphify-postgresql.serviceConfig.ExecStart}" != ""
                touch "$out"
              ''
            else
              pkgs.runCommand "infernix-graphify-nixos-module-unavailable" { } ''
                echo "Graphify input predates nixosModules.default; override or bump it to exercise this check." >&2
                touch "$out"
              '';

          workload-fabric = pkgs.runCommand "infernix-workload-fabric-check" { } ''
            test "${workloadFabricSample.config.services.infernix.workloadFabric.workerId}" = "atlas"
            case ${pkgs.lib.escapeShellArg (toString workloadFabricSample.config.systemd.services.infernix-workerd.serviceConfig.ExecStart)} in
              *"/bin/infernix-workerd --config"*" worker") ;;
              *) echo "workerd service does not run the worker command" >&2; exit 1 ;;
            esac
            requires='${builtins.toJSON workloadFabricSample.config.systemd.services.infernix-workerd.requires}'
            printf '%s' "$requires" | ${pkgs.jq}/bin/jq -e 'index("infernix-workload-migrate.service")'
            queues='${builtins.toJSON workloadFabricSample.config.services.infernix.workloadFabric.adapters.graphify.queues}'
            printf '%s' "$queues" | ${pkgs.jq}/bin/jq -e '.[0] == "code" and .[1] == "semantic"'
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

          model-catalog = pkgs.runCommand "infernix-model-catalog-check" { } ''
            test "${renderedLlamaSwapModels.qwen3-vl-8b.repo}" = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
            test "${builtins.elemAt renderedLlamaSwapModels.qwen3-vl-8b.extraArgs 0}" = "--mmproj /models/mmproj.gguf"
            test "${renderedFleetModels.qwen3-vl-8b.name}" = "qwen3-vl-8b"
            test "${builtins.elemAt renderedFleetModels.qwen3-vl-8b.capabilities 0}" = "chat"
            test "${renderedHmModels.vlm.name}" = "qwen3-vl-8b"
            test "${toString renderedHmModels.vlm.ctxSize}" = "4096"
            touch "$out"
          '';
        });
    };
}
