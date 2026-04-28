{
  config,
  infernixEmbr,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) literalExpression mkEnableOption mkIf mkOption optionals types;
  cfg = config.services.infernix.embr;
  qdrantCfg = config.services.infernix.qdrant;
  ollamaCfg = config.services.infernix.ollama;
  system = pkgs.stdenv.hostPlatform.system;

  localhostOf = host:
    if builtins.elem host [
      "0.0.0.0"
      "::"
      "[::]"
    ]
    then "127.0.0.1"
    else host;

  namedVectorSubmodule = types.submodule {
    options = {
      name = mkOption {
        type = types.str;
        example = "code";
        description = "Qdrant named vector to write and query.";
      };
      model = mkOption {
        type = types.str;
        example = "qwen3-embedding:8b";
        description = "Embedding model used for this named vector.";
      };
      dim = mkOption {
        type = types.ints.positive;
        example = 4096;
        description = "Vector dimension for this named vector.";
      };
    };
  };
in {
  options.services.infernix.embr = {
    enable = mkEnableOption "Embr project-code indexing via Infernix-owned Qdrant and Ollama services";

    package = mkOption {
      type = types.package;
      default = infernixEmbr.packages.${system}.embr;
      defaultText = literalExpression "inputs.infernix.packages.<system>.embr";
      description = "embr package to use for indexing.";
    };

    user = mkOption {
      type = types.str;
      default = "embr";
      description = "System user the indexer runs as.";
    };

    group = mkOption {
      type = types.str;
      default = "embr";
      description = "Primary group for the indexer user.";
    };

    projectsRoot = mkOption {
      type = types.path;
      example = "/mnt/atlas-projects";
      description = "Base directory containing project sub-directories.";
    };

    projects = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["canix" "infernix"];
      description = "Sub-directory names under projectsRoot to index.";
    };

    chunking = {
      maxLines = mkOption {
        type = types.ints.positive;
        default = 80;
        description = "Maximum lines per chunk.";
      };

      overlap = mkOption {
        type = types.ints.unsigned;
        default = 10;
        description = "Overlap between adjacent chunks, in lines.";
      };

      extensions = mkOption {
        type = types.listOf types.str;
        default = [
          "nix"
          "rs"
          "ts"
          "tsx"
          "js"
          "jsx"
          "py"
          "go"
          "md"
          "txt"
          "toml"
          "yaml"
          "yml"
          "json"
          "sh"
          "nu"
          "html"
          "css"
          "scss"
          "sql"
          "proto"
          "graphql"
          "lua"
          "c"
          "h"
          "cpp"
          "hpp"
        ];
        description = "File extensions considered source code.";
      };

      maxFileBytes = mkOption {
        type = types.ints.positive;
        default = 500000;
        description = "Skip files larger than this size in bytes.";
      };
    };

    watch = {
      intervalSecs = mkOption {
        type = types.ints.positive;
        default = 30;
        description = "Watch poll interval in seconds.";
      };

      forcePoll = mkOption {
        type = types.bool;
        default = false;
        description = "Force polling mode even on local filesystems.";
      };
    };

    requires = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["qdrant.service" "ollama.service"];
      description = "Additional systemd units that embr should wait for.";
    };

    qdrant = {
      useInfernixService = mkOption {
        type = types.bool;
        default = qdrantCfg.enable;
        defaultText = literalExpression "config.services.infernix.qdrant.enable";
        description = ''
          Whether to derive the Qdrant endpoint from the host's
          `services.infernix.qdrant` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.qdrant.useInfernixService
          then "http://${localhostOf qdrantCfg.host}:${toString qdrantCfg.httpPort}"
          else "http://127.0.0.1:6333";
        description = "Qdrant HTTP endpoint embr should use.";
      };

      collection = mkOption {
        type = types.str;
        default = "projects";
        description = "Qdrant collection name to create or reuse.";
      };
    };

    embedding = {
      useInfernixService = mkOption {
        type = types.bool;
        default = ollamaCfg.enable;
        defaultText = literalExpression "config.services.infernix.ollama.enable";
        description = ''
          Whether to derive the embedding endpoint from the host's
          `services.infernix.ollama` module.
        '';
      };

      url = mkOption {
        type = types.str;
        default =
          if cfg.embedding.useInfernixService
          then "http://${localhostOf ollamaCfg.host}:${toString ollamaCfg.port}"
          else "http://127.0.0.1:11434";
        description = "Ollama endpoint embr should use for query-time and indexing embeddings.";
      };

      vectors = mkOption {
        type = types.listOf namedVectorSubmodule;
        default = [];
        example = [
          {
            name = "code";
            model = "qwen3-embedding:8b";
            dim = 4096;
          }
        ];
        description = "Named embedding vectors to write into the Qdrant collection.";
      };
    };
  };

  config = mkIf cfg.enable {
    assertions =
      optionals cfg.qdrant.useInfernixService [
        {
          assertion = qdrantCfg.enable;
          message = ''
            services.infernix.embr.qdrant.useInfernixService = true requires
            services.infernix.qdrant.enable = true.
          '';
        }
      ]
      ++ optionals cfg.embedding.useInfernixService [
        {
          assertion = ollamaCfg.enable;
          message = ''
            services.infernix.embr.embedding.useInfernixService = true requires
            services.infernix.ollama.enable = true.
          '';
        }
      ];

    services.embr = {
      enable = true;
      inherit (cfg) package user group projectsRoot projects requires;
      chunking = {
        inherit (cfg.chunking) maxLines overlap extensions maxFileBytes;
      };
      watch = {
        inherit (cfg.watch) intervalSecs forcePoll;
      };
      qdrant = {
        inherit (cfg.qdrant) url collection;
      };
      embedding = {
        inherit (cfg.embedding) url vectors;
      };
    };
  };
}
