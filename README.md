# nix-infernis

Declarative NixOS and home-manager modules for self-hosted AI/ML inference.

`infernis` wraps the moving parts of a local model-serving stack — Ollama,
llama-swap (llama.cpp), Qdrant, SurrealDB, and [yeeHaw](https://codeberg.org/caniko/yeehaw)
— behind a single `services.infernis.*` namespace with a shared GPU vendor
abstraction so switching between AMD/ROCm, NVIDIA/CUDA, and CPU is a one-line
change.

## What it provides

NixOS modules (`nixosModules.default`):

- **`services.infernis.gpu`** — vendor abstraction (`amd` / `nvidia` / `cpu`)
  consumed by every other module. Handles `HSA_OVERRIDE_GFX_VERSION`, visible
  device masking, and a `pkgs` passthrough so you bring your own
  `rocmSupport`/`cudaSupport`-enabled nixpkgs instantiation.
- **`services.infernis.ollama`** — wraps `services.ollama` with GPU-aware
  package selection, host/port options, model preloading, and firewall rules.
- **`services.infernis.llama-swap`** — multi-model orchestrator over
  `llama-server`. Supports speculative decoding (draft models), automatic
  download of GGUF files from HuggingFace via a `infernis-download` systemd
  oneshot, per-model TTL and extra CLI args, flash-attention all-quants, and
  CPU microarchitecture tuning.
- **`services.infernis.qdrant`** — vector database with HTTP/gRPC ports,
  storage and snapshot path options, and firewall handling.
- **`services.infernis.surrealdb`** — SurrealDB multi-model database with
  backend selection via `backend = "surrealkv" | "rocksdb" | "memory"` with
  SurrealKV as the default, optional raw `dbPath` override, structured root
  auth options, extra CLI flags, and firewall handling.
- **`services.infernis.mnemo`** — installs the Mnemo CLI/MCP/hooks bundle,
  writes `/etc/mnemo/config.toml`, and can follow infernis-managed SurrealDB
  and Qdrant automatically.

### Bleeding-edge inference packages

infernis pulls `ollama`, `llama-cpp`, and `llama-swap` from `nixpkgs/master`
automatically via a dedicated `nixpkgs-bleeding` flake input, so you get the
newest inference features days-to-weeks ahead of `nixos-unstable`. This is
wired through `nixosModules.default` — consumers do **not** need to add an
overlay or a second input. `qdrant`, `curl`, and everything else still honor
the consumer's own `nixos-unstable`-tracking pkgs passed into
`services.infernis.gpu.pkgs`.

Caveat: master revs are not channel snapshots, so `ollama-rocm` /
`ollama-cuda` are **not** in `cache.nixos.org` or `cuda-maintainers.cachix.org`.
Expect a local ollama rebuild on each `nix flake update nixpkgs-bleeding`
(Go build, a few minutes). `llama-cpp` with `rocmSupport` / `cudaSupport` was
already built locally on every channel regardless — master costs nothing
extra there.

Home-manager modules (`homeModules.default`):

- **`services.infernis.endpoints`** — the central abstraction. You declare
  each reachable model backend once (type, URL, models with context size and
  optional role hints) and every other HM module consumes it.
- **`services.infernis.ollama` aliases** — auto-generated `ollama-load-<model>`
  and `ollama-unload-<model>` shell aliases derived from your ollama
  endpoints.
- **`services.infernis.yeehaw`** — auto-generates yeeHaw "steeds" from
  `endpoints`. Exposed as the read-only option
  `services.infernis.yeehaw.generatedSteeds` rather than writing directly to
  `programs.yh.steeds`: because home-manager's `sharedModules` apply to every
  user, writing to `programs.yh.steeds` under `mkIf` still triggers
  type-checking for users who don't import yeeHaw's HM module. Wire it in
  yourself with one line per user — see the snippet below.

Additional opt-in Home Manager modules:

- **`homeModules.goose`** — writes `programs.goose.*` from the generated
  `services.infernis.goose.*` outputs for users that also import the Goose
  Home Manager module.
- **`homeModules.yeehaw`** — writes `programs.yh.steeds` from
  `services.infernis.yeehaw.generatedSteeds` for users that also import the
  yeeHaw Home Manager module.

## Quick start

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    home-manager.url = "github:nix-community/home-manager";
    nix-infernis.url = "codeberg:caniko/nix-infernis";
  };

  outputs = {nixpkgs, home-manager, nix-infernis, ...}: {
    nixosConfigurations.atlas = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        nix-infernis.nixosModules.default
        ({pkgs, ...}: {
          services.infernis.gpu = {
            vendor = "amd";
            pkgs = pkgs;          # pass a pkgs with rocmSupport = true
            amd.gfxVersion = "11.0.0";
          };

          services.infernis.ollama = {
            enable = true;
            loadModels = ["gemma4:31b-it-q4_K_M"];
          };

          services.infernis.llama-swap = {
            enable = true;
            modelsDir = "/var/lib/llama-models";
            hfTokenPath = "/run/secrets/hf-token";
            models.coder = {
              repo = "unsloth/Qwen3-Coder-Next-GGUF";
              file = "Qwen3-Coder-Next-Q4_K_M.gguf";
              ctxSize = 65536;
            };
          };

          services.infernis.qdrant.enable = true;
          services.infernis.surrealdb = {
            enable = true;
            # Optional: defaults to "surrealkv".
            backend = "surrealkv";
            auth = {
              enable = true;
              password = "replace-me";
            };
          };

          services.infernis.mnemo = {
            enable = true;
            settings.storage.cache_size_mb = 256;
          };
        })
      ];
    };

    homeConfigurations."can@atlas" = home-manager.lib.homeManagerConfiguration {
      modules = [
        nix-infernis.homeModules.default
        ({config, ...}: {
          services.infernis.endpoints = {
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
                role = "deep";
                blockingGroup = "local-gpu";
              };
            };
          };

          services.infernis.yeehaw.enable = true;

          # Wire the generated steeds into yeeHaw yourself. See note above.
          programs.yh.steeds = config.services.infernis.yeehaw.generatedSteeds;
        })
      ];
    };
  };
}
```

If you already import yeeHaw's Home Manager module, you can replace the final
manual wiring line with `nix-infernis.homeModules.yeehaw` in the module list.

When `services.infernis.mnemo.enable = true`, infernis writes
`/etc/mnemo/config.toml` and defaults to the host's
`services.infernis.surrealdb` and `services.infernis.qdrant` endpoints.
SurrealDB integration requires
`services.infernis.surrealdb.auth.enable = true` because Mnemo always
authenticates over WebSocket. To point Mnemo somewhere else, set
`services.infernis.mnemo.surrealdb.useInfernisService = false` and/or
`services.infernis.mnemo.qdrant.useInfernisService = false`, then override the
connection fields explicitly.

## GPU configuration

`services.infernis.gpu.pkgs` must be a nixpkgs instantiation with the
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

`services.infernis.llama-swap` ships a `infernis-download` systemd oneshot
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
