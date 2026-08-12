{lib}: let
  codexProviderPort = 3967;
  deepseekModel = id: {
    inherit id;
    name = id;
    reasoning = true;
    tool_call = true;
    limit = {
      context = 750000;
      output = 65536;
    };
    compaction = {
      auto = true;
      prune = true;
    };
  };

  model = {
    id,
    name ? id,
    context,
    output,
    reasoning ? false,
  }: {
    inherit id name reasoning;
    tool_call = true;
    limit = {
      context = context;
      output = output;
    };
  };
in {
  providers = {
    deepseek = {
      name = "DeepSeek";
      protocol = "openai-chat";
      baseUrl = "https://api.deepseek.com";
      apiKeyEnv = "DEEPSEEK_API_KEY";
      models = {
        deepseek-v4-flash = deepseekModel "deepseek-v4-flash";
        deepseek-v4-pro = deepseekModel "deepseek-v4-pro";
      };
    };

    xiaomi = {
      name = "Xiaomi MiMo";
      protocol = "openai-chat";
      baseUrl = "https://token-plan-ams.xiaomimimo.com/v1";
      apiKeyEnv = "XIAOMI_PLATFORM_TOKEN";
      models = {
        mimo-v2-pro = model {
          id = "mimo-v2-pro";
          context = 1000000;
          output = 128000;
        };
        "mimo-v2.5-pro" = model {
          id = "mimo-v2.5-pro";
          context = 1000000;
          output = 128000;
        };
        mimo-v2-flash = model {
          id = "mimo-v2-flash";
          context = 1000000;
          output = 128000;
        };
        mimo-v2-omni = model {
          id = "mimo-v2-omni";
          context = 256000;
          output = 128000;
        };
      };
    };

    opencode = {
      name = "OpenCode Zen";
      protocol = "openai-chat";
      baseUrl = "https://opencode.ai/zen/v1";
      apiKey = "public";
      models = {
        deepseek-v4-flash-free = deepseekModel "deepseek-v4-flash-free";
      };
    };

    opencode-go = {
      name = "OpenCode Go";
      protocol = "openai-chat";
      baseUrl = "https://opencode.ai/zen/go/v1";
      apiKeyEnv = "OPENCODE_API_KEY";
      models = {
        "glm-5.2" = model {
          id = "glm-5.2";
          name = "GLM-5.2";
          context = 131072;
          output = 16384;
        };
      };
    };

    gmi = {
      name = "GMI Cloud";
      protocol = "openai-chat";
      baseUrl = "https://api.gmi-serving.com/v1";
      apiKeyEnv = "GMI_CLOUD_API_KEY";
      models = {
        "zai-org/GLM-5.2-FP8" = model {
          id = "zai-org/GLM-5.2-FP8";
          context = 131072;
          output = 16384;
        };
      };
    };

    codex = {
      name = "Codex CLI";
      protocol = "openai-chat";
      baseUrl = "http://127.0.0.1:${toString codexProviderPort}/v1";
      apiKey = "infernix-local";
      models = {
        default = model {
          id = "default";
          name = "Codex (configured default)";
          context = 114688;
          output = 32768;
        };
        "gpt-5.5" = model {
          id = "gpt-5.5";
          context = 114688;
          output = 32768;
        };
      };
    };
  };

  routes = {
    default = "opencode-go,glm-5.2";
    background = "deepseek,deepseek-v4-flash";
    think = "deepseek,deepseek-v4-flash";
    longContext = "xiaomi,mimo-v2.5-pro";
    longContextThreshold = 60000;
  };

  inherit codexProviderPort;
}
