# infernix

Declarative NixOS and home-manager modules for self-hosted AI/ML inference.

`infernix` wraps the moving parts of a local model-serving stack — Ollama,
llama-swap (llama.cpp), Qdrant, and SurrealDB — behind a single
`services.infernix.*` namespace with a shared GPU vendor abstraction so
switching between AMD/ROCm, NVIDIA/CUDA, and CPU is a one-line change. Its
Home Manager modules can also translate local Infernix-backed endpoints into
Goose defaults/providers and [yeeHaw](https://codeberg.org/caniko/yeehaw)
steeds.

## What it provides

NixOS modules (`nixosModules.default`):

- **`services.infernix.gpu`** — vendor abstraction (`amd` / `nvidia` / `cpu`)
  consumed by every other module. Handles `HSA_OVERRIDE_GFX_VERSION`, visible
  device masking, and a `pkgs` passthrough so you bring your own
  `rocmSupport`/`cudaSupport`-enabled nixpkgs instantiation.
- **`services.infernix.ollama`** — wraps `services.ollama` with GPU-aware
  package selection, host/port options, model preloading, and firewall rules.
- **`services.infernix.llama-swap`** — multi-model orchestrator over
  `llama-server`. Supports speculative decoding (draft models), automatic
  download of GGUF files from HuggingFace via a `infernix-download` systemd
  oneshot, per-model TTL and extra CLI args, flash-attention all-quants, and
  CPU microarchitecture tuning.
- **`services.infernix.qdrant`** — vector database with HTTP/gRPC ports,
  storage and snapshot path options, and firewall handling.
- **`services.infernix.surrealdb`** — SurrealDB multi-model database with
  backend selection via `backend = "surrealkv" | "rocksdb" | "memory"` with
  SurrealKV as the default, optional raw `dbPath` override, structured root
  auth options, extra CLI flags, and firewall handling.
- **`services.infernix.mnemo`** — installs the Mnemo CLI/MCP/hooks bundle,
  writes `/etc/mnemo/config.toml`, and can follow infernix-managed SurrealDB
  and Qdrant automatically.

### Inference packages

infernix uses a single `nixos-unstable` nixpkgs input for all packages,
including `ollama`, `llama-cpp`, and `llama-swap`. GPU-aware packages are
re-instantiated with the consumer's own `services.infernix.gpu.pkgs` config so
they keep `rocmSupport`, `cudaSupport`, `allowUnfree`, and similar settings
without introducing a second nixpkgs channel.

Home-manager modules (`homeModules.default`):

- **`services.infernix.endpoints`** — the central abstraction. You declare
  each reachable local model backend once (type, URL, models with context size
  and optional contention metadata) and every other HM module consumes it.
- **`services.infernix.ollama` aliases** — auto-generated `ollama-load-<model>`
  and `ollama-unload-<model>` shell aliases derived from your ollama
  endpoints.
- **`services.infernix.goose`** — generates Goose custom provider definitions
  and default settings from local Infernix-backed endpoints. Exposed as
  read-only outputs so they can be wired into `programs.goose.*`.
- **`services.infernix.yeehaw`** — auto-generates yeeHaw "steeds" from
  local Infernix-backed `endpoints`. Exposed as the read-only option
  `services.infernix.yeehaw.generatedSteeds` rather than writing directly to
  `programs.yh.steeds`: because home-manager's `sharedModules` apply to every
  user, writing to `programs.yh.steeds` under `mkIf` still triggers
  type-checking for users who don't import yeeHaw's HM module. Wire it in
  yourself with one line per user — see the snippet below.

Additional opt-in Home Manager modules:

- **`homeModules.goose`** — writes `programs.goose.*` from the generated
  `services.infernix.goose.*` outputs for users that also import a Goose
  Home Manager module.
- **`homeModules.yeehaw`** — writes `programs.yh.steeds` from
  `services.infernix.yeehaw.generatedSteeds` for users that also import the
  yeeHaw Home Manager module.

## Quick start

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager.url = "github:nix-community/home-manager";
    infernix.url = "codeberg:caniko/infernix";
  };

  outputs = {nixpkgs, home-manager, infernix, ...}: {
    nixosConfigurations.atlas = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        infernix.nixosModules.default
        ({pkgs, ...}: {
          services.infernix.gpu = {
            vendor = "amd";
            pkgs = pkgs;          # pass a pkgs with rocmSupport = true
            amd.gfxVersion = "11.0.0";
          };

          services.infernix.ollama = {
            enable = true;
            loadModels = ["gemma4:31b-it-q4_K_M"];
          };

          services.infernix.llama-swap = {
            enable = true;
            modelsDir = "/var/lib/llama-models";
            hfTokenPath = "/run/secrets/hf-token";
            models.coder = {
              repo = "unsloth/Qwen3-Coder-Next-GGUF";
              file = "Qwen3-Coder-Next-Q4_K_M.gguf";
              ctxSize = 65536;
            };
          };

          services.infernix.qdrant.enable = true;
          services.infernix.surrealdb = {
            enable = true;
            # Optional: defaults to "surrealkv".
            backend = "surrealkv";
            auth = {
              enable = true;
              password = "replace-me";
            };
          };

          services.infernix.mnemo = {
            enable = true;
            settings.storage.cache_size_mb = 256;
          };
        })
      ];
    };

    homeConfigurations."can@atlas" = home-manager.lib.homeManagerConfiguration {
      modules = [
        infernix.homeModules.default
        infernix.homeModules.yeehaw
        ({config, ...}: {
          services.infernix.endpoints = {
            local-ollama = {
              type = "ollama";
              url = "http://localhost:11434";
              models.gemma4.name = "gemma4:31b-it-q4_K_M";
            };

            local-llama-swap = {
              type = "llama-swap";
              url = "http://localhost:8013";
              models.coder = {
                name = "qwen3-coder-next";
                ctxSize = 65536;
                blockingGroup = "local-gpu";
              };
            };
          };

          services.infernix.goose = {
            enable = true;
            defaultEndpoint = "local-llama-swap";
            defaultModel = "coder";
          };

          services.infernix.yeehaw.enable = true;
        })
      ];
    };
  };
}
```

If you already import Goose's Home Manager module, add
`infernix.homeModules.goose` to the module list to write
`programs.goose.*` from `services.infernix.goose.*`.

If you already import yeeHaw's Home Manager module, add
`infernix.homeModules.yeehaw` to the module list to write
`programs.yh.steeds` from `services.infernix.yeehaw.generatedSteeds`.

When `services.infernix.mnemo.enable = true`, infernix writes
`/etc/mnemo/config.toml` and defaults to the host's
`services.infernix.surrealdb` and `services.infernix.qdrant` endpoints.
SurrealDB integration requires
`services.infernix.surrealdb.auth.enable = true` because Mnemo always
authenticates over WebSocket. To point Mnemo somewhere else, set
`services.infernix.mnemo.surrealdb.useInfernixService = false` and/or
`services.infernix.mnemo.qdrant.useInfernixService = false`, then override the
connection fields explicitly.

For local Mnemo development, keep Infernix's default input remote and override
it per command instead of hard-coding a machine-local path:

```bash
nix build .#default --override-input mnemo path:/path/to/mnemo
```

## GPU configuration

`services.infernix.gpu.pkgs` must be a nixpkgs instantiation with the
appropriate GPU support flag already set. In practice that means either
building a dedicated `import nixpkgs { config.rocmSupport = true; ... }` in
your flake, or pulling `pkgsPrimaryGpu` from
[canix](https://codeberg.org/caniko/canix). For AMD you will usually also set
`amd.gfxVersion` (e.g. `"11.0.0"` for gfx1100) and optionally
`amd.cpuArch` to drop a `-march=znver4`-style flag into the llama.cpp build.

The `visibleDevices` option masks devices at the systemd unit level via
`ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`, so you can keep a second GPU
free for other workloads.

## Model downloads

`services.infernix.llama-swap` ships a `infernix-download` systemd oneshot
that runs before `llama-swap.service` and `curl`s every referenced `repo`/
`file` pair from HuggingFace into `modelsDir` if it's not already there.
Gated repos work by setting `hfTokenPath` to a file containing a HuggingFace
access token — typically a sops-nix or agenix secret. Both main models and
speculative-decoding draft models are downloaded the same way.

## Status

Early. The module API may still change without deprecation warnings.

## CI

Woodpecker CI on Codeberg runs `nix flake check` on every push and pull request to verify that all module definitions evaluate correctly.

## License

Dual-licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  https://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  https://www.apache.org/licenses/LICENSE-2.0)

at your option.
