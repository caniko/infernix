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

    rerankerModel = mkOption {
      type = types.str;
      default = "jina-reranker-v3";
      description = "Reranker model name sent by Pink Raven.";
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
