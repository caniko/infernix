{
  config,
  infernixHeadroom ? null,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkEnableOption mkOption types;
  cfg = config.services.infernix.headroom;
  instances = lib.filterAttrs (_: instance: instance.enable) cfg.instances;
  baseUrl = instance: "http://127.0.0.1:${toString instance.port}";
  commonEnvironment = {
    HEADROOM_BEACON = "off";
    HEADROOM_OFFLINE = "1";
    HEADROOM_CCR_BACKEND = "sqlite";
    HEADROOM_CCR_SQLITE_PATH = "${cfg.stateDirectory}/ccr.sqlite";
    HEADROOM_CCR_TTL_SECONDS = toString cfg.retentionSeconds;
    HEADROOM_TOOL_SEARCH = "0";
    HEADROOM_OUTPUT_SHAPER = "off";
    HEADROOM_DISABLE_KOMPRESS = "1";
    HEADROOM_DISABLE_KOMPRESS_FALLBACK = "1";
    HEADROOM_KOMPRESS_WARMUP = "0";
    HF_HUB_OFFLINE = "1";
    LITELLM_LOCAL_MODEL_COST_MAP = "True";
    DO_NOT_TRACK = "1";
  };
  command = instance:
    lib.escapeShellArgs ([
        (lib.getExe cfg.package)
        "proxy"
        "--host"
        "127.0.0.1"
        "--port"
        (toString instance.port)
        "--mode"
        instance.mode
        "--no-cache"
        "--no-learn"
        "--no-telemetry"
      ]
      ++ lib.optionals (instance.openaiUrl != null) ["--openai-api-url" instance.openaiUrl]
      ++ lib.optionals (instance.anthropicUrl != null) ["--anthropic-api-url" instance.anthropicUrl]);
in {
  options.services.infernix.headroom = {
    enable = mkEnableOption "per-user Headroom context compression proxies";
    package = mkOption {
      type = types.nullOr types.package;
      default = infernixHeadroom;
      description = "Pinned Headroom proxy package; supplied by the Infernix flake module.";
    };
    stateDirectory = mkOption {
      type = types.str;
      default = "${config.xdg.stateHome}/infernix-headroom";
      description = "Private proxy state and shared SQLite compression store.";
    };
    retentionSeconds = mkOption {
      type = types.ints.positive;
      default = 86400;
      description = "Compression retrieval retention, including across proxy restarts.";
    };
    instances = mkOption {
      default = {};
      type = types.attrsOf (types.submodule {
        options = {
          enable = mkOption {
            type = types.bool;
            default = true;
          };
          port = mkOption {type = types.port;};
          openaiUrl = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "OpenAI-compatible upstream base URL, including its API prefix.";
          };
          anthropicUrl = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Anthropic upstream origin (without /v1/messages).";
          };
          mode = mkOption {
            type = types.enum ["cache" "token"];
            default = "cache";
          };
          requires = mkOption {
            type = types.listOf types.str;
            default = [];
          };
          environment = mkOption {
            type = types.attrsOf types.str;
            default = {};
            description = "Non-secret proxy environment overrides.";
          };
          environmentFile = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Optional runtime environment file; never a store-resident credential.";
          };
          memoryMax = mkOption {
            type = types.str;
            default = "768M";
          };
        };
      });
    };
    endpoints = mkOption {
      type = types.attrsOf (types.submodule {
        options = {
          url = mkOption {type = types.str;};
          healthUrl = mkOption {type = types.str;};
          unit = mkOption {type = types.str;};
        };
      });
      readOnly = true;
      default =
        lib.mapAttrs (name: instance: {
          url = baseUrl instance;
          healthUrl = "${baseUrl instance}/health";
          unit = "infernix-headroom-${name}.service";
        })
        instances;
    };
    units = mkOption {
      type = types.listOf types.str;
      readOnly = true;
      default = map (endpoint: endpoint.unit) (builtins.attrValues cfg.endpoints);
    };
    mcp = {
      enable = mkEnableOption "Fleetix-managed Headroom compression retrieval";
      instance = mkOption {
        type = types.str;
        default = "codex";
      };
      harnesses = mkOption {
        type = types.listOf (types.enum ["opencode" "codex" "claude"]);
        default = ["opencode" "codex" "claude"];
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [
      {
        assertion = cfg.package != null;
        message = "Headroom requires its Infernix package or an explicit package.";
      }
      {
        assertion = lib.hasPrefix "/" cfg.stateDirectory;
        message = "Headroom stateDirectory must be absolute.";
      }
      {
        assertion = builtins.length (lib.unique (map (i: i.port) (builtins.attrValues instances))) == builtins.length (builtins.attrNames instances);
        message = "Headroom proxy ports must be unique.";
      }
      {
        assertion = lib.all (name: builtins.match "[A-Za-z0-9_-]+" name != null) (builtins.attrNames instances);
        message = "Headroom instance names must be safe systemd unit components.";
      }
      {
        assertion = !cfg.mcp.enable || builtins.hasAttr cfg.mcp.instance instances;
        message = "Headroom MCP requires an enabled retrieval proxy instance.";
      }
    ];
    home.packages = [cfg.package];
    systemd.user.services = lib.mapAttrs' (name: instance:
      lib.nameValuePair "infernix-headroom-${name}" {
        Unit = {
          Description = "Headroom context compression (${name})";
          Requires = instance.requires;
          After = ["network.target"] ++ instance.requires;
        };
        Service =
          {
            ExecStartPre = "${pkgs.coreutils}/bin/install -d -m 0700 ${lib.escapeShellArg cfg.stateDirectory} ${lib.escapeShellArg "${cfg.stateDirectory}/${name}"}";
            ExecStart = command instance;
            ExecStartPost = "${lib.getExe pkgs.curl} --fail --silent --show-error --retry 30 --retry-connrefused --retry-delay 1 --retry-max-time 60 ${baseUrl instance}/health";
            Environment = lib.mapAttrsToList (key: value: "${key}=${value}") (commonEnvironment
              // {
                HEADROOM_WORKSPACE_DIR = "${cfg.stateDirectory}/${name}";
              }
              // instance.environment);
            UMask = "0077";
            NoNewPrivileges = true;
            PrivateTmp = true;
            MemoryAccounting = true;
            MemoryMax = instance.memoryMax;
            MemorySwapMax = "0";
            Restart = "on-failure";
            RestartSec = "2s";
            TimeoutStartSec = "90s";
            TimeoutStopSec = "15s";
            KillMode = "control-group";
          }
          // lib.optionalAttrs (instance.environmentFile != null) {EnvironmentFile = instance.environmentFile;};
        Install.WantedBy = ["default.target"];
      })
    instances;
    fleetix.mcp = lib.mkIf cfg.mcp.enable {
      enable = lib.mkDefault true;
      # Above the option's false default, below consumer mkDefault policy.
      harnesses = lib.genAttrs cfg.mcp.harnesses (_: {enable = lib.mkOverride 1100 true;});
      servers.headroom = {
        command = lib.getExe cfg.package;
        args = ["mcp" "serve" "--proxy-url" cfg.endpoints.${cfg.mcp.instance}.url];
        env = commonEnvironment;
        inherit (cfg.mcp) harnesses;
      };
    };
  };
}
