{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption mkIf types;
  cfg = config.services.infernis.embedder;

  embedderPkg = pkgs.callPackage ../../packages/embedder.nix {};

  # Generated JSON config handed to the nushell indexer at runtime.
  indexerConfig = pkgs.writeText "infernis-embedder.json" (builtins.toJSON {
    qdrant = {
      inherit (cfg.qdrant) url collection;
    };
    embedding = {
      inherit (cfg.embedding) url textModel codeModel vectorDim;
    };
    inherit (cfg) projectsRoot projects;
    chunking = {
      inherit (cfg.chunking) maxLines overlap extensions;
    };
  });
in {
  options.services.infernis.embedder = {
    enable = mkEnableOption "declarative project RAG indexer";

    package = mkOption {
      type = types.package;
      default = embedderPkg;
      description = "infernis-embedder indexer package.";
    };

    qdrant = {
      url = mkOption {
        type = types.str;
        default = "http://localhost:6333";
        description = "Qdrant base URL.";
      };
      collection = mkOption {
        type = types.str;
        default = "projects";
        description = "Qdrant collection name. Created on first run if absent.";
      };
    };

    embedding = {
      url = mkOption {
        type = types.str;
        default = "http://localhost:11434";
        description = "Ollama base URL (used for /api/embed).";
      };
      textModel = mkOption {
        type = types.str;
        default = "nomic-embed-text";
        description = "Ollama model name for the 'text' named vector.";
      };
      codeModel = mkOption {
        type = types.str;
        default = "nomic-embed-code";
        description = "Ollama model name for the 'code' named vector.";
      };
      vectorDim = mkOption {
        type = types.int;
        default = 768;
        description = "Vector dimension for both named vectors (must match model output).";
      };
    };

    projectsRoot = mkOption {
      type = types.path;
      example = "/mnt/atlas-projects";
      description = "Base directory containing project subdirectories.";
    };

    projects = mkOption {
      type = types.listOf types.str;
      default = [];
      example = ["canix" "nix-infernis"];
      description = "Subdirectory names under projectsRoot to index.";
    };

    chunking = {
      maxLines = mkOption {
        type = types.int;
        default = 80;
        description = "Maximum lines per chunk.";
      };
      overlap = mkOption {
        type = types.int;
        default = 10;
        description = "Overlapping lines between adjacent chunks.";
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
        description = "File extensions (without leading dot) considered source.";
      };
    };

    schedule = mkOption {
      type = types.str;
      default = "hourly";
      example = "*-*-* 02:00:00";
      description = "systemd OnCalendar expression for the indexer timer.";
    };

    user = mkOption {
      type = types.str;
      default = "infernis-embedder";
      description = "System user the indexer runs as.";
    };
  };

  config = mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.user;
      description = "infernis-embedder indexer";
    };
    users.groups.${cfg.user} = {};

    systemd.services.infernis-embedder = {
      description = "infernis RAG project indexer";
      after = ["network-online.target"];
      wants = ["network-online.target"];
      # Best-effort ordering: run after local services if they exist on this host.
      # If either is absent, systemd treats the After= as a no-op.
      unitConfig.After = ["qdrant.service" "ollama.service"];

      serviceConfig = {
        Type = "oneshot";
        User = cfg.user;
        Group = cfg.user;
        ExecStart = "${lib.getExe cfg.package} ${indexerConfig}";
        # Hardening
        NoNewPrivileges = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        PrivateTmp = true;
        ReadOnlyPaths = [cfg.projectsRoot];
        # Timeout generous — first run embeds everything and can be slow.
        TimeoutStartSec = "2h";
      };
    };

    systemd.timers.infernis-embedder = {
      description = "infernis RAG project indexer (schedule)";
      wantedBy = ["timers.target"];
      timerConfig = {
        OnCalendar = cfg.schedule;
        Persistent = true;
        RandomizedDelaySec = "5m";
      };
    };
  };
}
