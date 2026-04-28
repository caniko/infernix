# Endpoint declarations — the central abstraction for local Infernix-backed
# model endpoints that other HM modules consume.
{lib, ...}: let
  inherit (lib) mkOption types;

  endpointSubmodule = types.submodule {
    options = {
      type = mkOption {
        type = types.enum ["ollama" "llama-swap"];
        description = "Backend type.";
      };

      url = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "http://localhost:11434";
        description = ''
          Base URL to reach this endpoint.
        '';
      };

      containerUrl = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "http://host.docker.internal:8013/v1";
        description = ''
          URL reachable from inside Docker containers (for yeeHaw).
          If null, derived from url by replacing localhost with host.docker.internal.
        '';
      };

      models = mkOption {
        type = types.attrsOf (types.submodule {
          options = {
            name = mkOption {
              type = types.str;
              example = "qwen3-coder-next";
              description = "Model identifier passed to the backend.";
            };

            ctxSize = mkOption {
              type = types.nullOr types.int;
              default = null;
              example = 65536;
              description = "Context window size in tokens.";
            };

            blockingGroup = mkOption {
              type = types.nullOr types.str;
              default = null;
              example = "local-gpu";
              description = "GPU contention group (steeds in the same group share a physical GPU).";
            };
          };
        });
        default = {};
        description = "Models available on this endpoint.";
      };
    };
  };
in {
  options.services.infernix.endpoints = mkOption {
    type = types.attrsOf endpointSubmodule;
    default = {};
    description = ''
      Named local model endpoints. Each endpoint describes a reachable Infernix-
      backed model backend that other Infernix HM modules and downstream config
      can consume to generate local integrations.
    '';
    example = {
      local-ollama = {
        type = "ollama";
        url = "http://localhost:11434";
        models.gemma4 = {
          name = "gemma4:31b-it-q4_K_M";
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
        models.cerebral = {
          name = "gemma4-31b";
          ctxSize = 32768;
          blockingGroup = "local-gpu";
        };
      };
    };
  };
}
