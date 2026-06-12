{
  config,
  lib,
  ...
}: let
  inherit (lib) mkEnableOption mkIf mkOption types;
  cfg = config.services.infernix.workloads.pinkRaven;
in {
  options.services.infernix.workloads.pinkRaven = {
    enable = mkEnableOption "Pink Raven ML logistics managed by Infernix";

    lbUrl = mkOption {
      type = types.str;
      default = "http://192.168.178.31:8014";
      description = "Base URL of the Infernix OpenAI-compatible load balancer.";
    };

    embeddingModel = mkOption {
      type = types.str;
      default = "qwen3-embedding-8b";
      description = "Embedding model name sent by Pink Raven.";
    };

    embeddingDim = mkOption {
      type = types.int;
      default = 4096;
      description = "Expected Pink Raven embedding vector dimension.";
    };

    embeddingTimeoutMs = mkOption {
      type = types.ints.positive;
      default = 180000;
      description = ''
        Pink Raven HTTP embedding request timeout in milliseconds. This should
        be long enough for a cold llama-swap backend to load the embedding
        model on first use.
      '';
    };

    rerankerModel = mkOption {
      type = types.str;
      default = "bge-reranker-v2-m3";
      description = ''
        Reranker model name sent by Pink Raven. The default is Apache-2.0
        licensed; jina-reranker-v3 is CC-BY-NC-4.0 and must not be used for
        commercial self-hosting.
      '';
    };

    rerankerBatchSize = mkOption {
      type = types.ints.positive;
      default = 64;
      description = "Maximum number of candidates per Pink Raven rerank request.";
    };

    captionModel = mkOption {
      type = types.str;
      default = "qwen3-vl-8b";
      description = "Caption model name sent by Pink Raven.";
    };

    region = mkOption {
      type = types.enum ["eu" "ch" "us" "unrestricted"];
      default = "unrestricted";
      description = "Pink Raven data-residency tag for Infernix ML endpoints.";
    };
  };

  config = mkIf cfg.enable {
    services.pink-raven = {
      embeddingBackend = "http";
      embeddingBackends = [
        {
          url = cfg.lbUrl;
          region = cfg.region;
        }
      ];
      embeddingLbStrategy = "least-in-flight";
      embeddingModel = cfg.embeddingModel;
      embeddingDim = cfg.embeddingDim;
      settings.PINK_RAVEN_EMBEDDING_TIMEOUT_MS = toString cfg.embeddingTimeoutMs;

      rerankerEnabled = true;
      rerankerUrl = "${cfg.lbUrl}/v1/rerank";
      rerankerRegion = cfg.region;
      rerankerModel = cfg.rerankerModel;
      rerankerBatchSize = cfg.rerankerBatchSize;

      captionUrl = cfg.lbUrl;
      captionRegion = cfg.region;
      captionModel = cfg.captionModel;
    };
  };
}
