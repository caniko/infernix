# Endpoint declarations — the central abstraction that other HM modules consume.
# An endpoint is a reachable model backend (ollama, llama-swap, claude, or codex).
{lib, ...}: let
  inherit (lib) mkOption types;

  endpointSubmodule = types.submodule {
    options = {
      type = mkOption {
        type = types.enum ["ollama" "llama-swap" "claude" "codex"];
        description = "Backend type.";
      };

      url = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "http://localhost:11434";
        description = ''
          Base URL to reach this endpoint.
          Not needed for claude or codex endpoints (they use external APIs/CLI auth).
        '';
      };

      containerUrl = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "http://host.docker.internal:8013/v1";
        description = ''
          URL reachable from inside Docker containers (for yeeHaw).
          If null, derived from url by replacing localhost with host.docker.internal.
          Not used for claude or codex endpoints.
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

            role = mkOption {
              type = types.nullOr (types.enum ["fast" "deep" "cerebral"]);
              default = null;
              description = ''
                Semantic role hint for consumer-side routing helpers.
                infernis does not auto-generate yeeHaw barns from this yet.
                - fast: quick completions, code execution
                - deep: complex coding, reasoning
                - cerebral: planning, analysis, sentinel
              '';
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
  options.services.infernis.endpoints = mkOption {
    type = types.attrsOf endpointSubmodule;
    default = {};
    description = ''
      Named model endpoints. Each endpoint describes a reachable model backend.
      Other infernis HM modules (ollama aliases, yeeHaw integration) consume these
      to auto-generate their configurations.
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
          role = "deep";
          blockingGroup = "local-gpu";
        };
        models.cerebral = {
          name = "gemma4-31b";
          ctxSize = 32768;
          role = "cerebral";
          blockingGroup = "local-gpu";
        };
      };
      claude = {
        type = "claude";
        models.opus = {
          name = "opus";
          role = "deep";
        };
        models.sonnet = {
          name = "sonnet";
          role = "fast";
        };
      };
      codex = {
        type = "codex";
        models.gpt-5-4 = {
          name = "gpt-5.4";
          role = "deep";
        };
        models.gpt-5-4-mini = {
          name = "gpt-5.4-mini";
          role = "fast";
        };
      };
    };
  };
}
