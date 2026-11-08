{
  pkgs,
  harborPy,
}: let
  # Tiktoken's cache key is SHA-1 of the URL; content is verified separately.
  tokenizerCache = pkgs.linkFarm "headroom-tokenizers" (pkgs.lib.mapAttrsToList (name: sha256: let
      url = "https://openaipublic.blob.core.windows.net/encodings/${name}.tiktoken";
    in {
      name = builtins.hashString "sha1" url;
      path = pkgs.fetchurl {inherit url sha256;};
    }) {
      cl100k_base = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
      o200k_base = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";
    });
  environment = harborPy.lib.mkUvVirtualEnv {
    inherit pkgs;
    python = pkgs.python312;
    name = "headroom-ai-0.40.0-env";
    workspaceRoot = ./.;
    dependencies = {headroom-ai = ["proxy"];};
    pyprojectOverrides = _final: prev:
      pkgs.lib.genAttrs ["onnxruntime" "sqlite-vec" "ast-grep-cli"] (name:
        prev.${name}.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pkgs.stdenv.cc.cc.lib];
        }))
      // {
        headroom-ai = prev.headroom-ai.overrideAttrs (old: {
          buildInputs = (old.buildInputs or []) ++ [pkgs.stdenv.cc.cc.lib];
          nativeBuildInputs = (old.nativeBuildInputs or []) ++ [pkgs.patch];
          postInstall =
            (old.postInstall or "")
            + ''
              patch --fuzz=0 --directory="$out/${pkgs.python312.sitePackages}" -p1 < ${./codex-routing.patch}
            '';
        });
      };
  };
in
  harborPy.lib.mkPythonApplicationPackage {
    inherit pkgs environment;
    name = "headroom-ai-0.40.0";
    scripts = ["headroom"];
    runtimeLibraryPackages = [pkgs.stdenv.cc.cc.lib pkgs.zlib];
    wrapperEnv = {
      HEADROOM_BEACON = "off";
      HF_HUB_OFFLINE = "1";
      LITELLM_LOCAL_MODEL_COST_MAP = "True";
      TIKTOKEN_CACHE_DIR = toString tokenizerCache;
    };
    passthru = {
      version = "0.40.0";
      inherit tokenizerCache;
    };
    meta = {
      description = "Context compression proxy for AI agents";
      homepage = "https://github.com/headroomlabs-ai/headroom";
      license = pkgs.lib.licenses.asl20;
      mainProgram = "headroom";
      platforms = pkgs.lib.platforms.linux;
    };
  }
