# Infernix vision

Infernix is a set of declarative NixOS and Home Manager modules for self-hosted
AI/ML inference on your own hardware. It puts the moving parts of a local
model-serving stack — Ollama, llama-swap over llama.cpp, ComfyUI, Qdrant,
SurrealDB, and Colibri profiles — behind one `services.infernix.*` namespace
with a shared GPU vendor abstraction, so switching between AMD/ROCm,
NVIDIA/CUDA, and CPU is a one-line change instead of a per-host rewrite.

## Problem

A local inference stack is not one service. It is several servers with
different model formats, vendor-specific environment variables, device masking,
model downloads that must be verified, and boot ordering that decides whether a
GPU is busy. Assembled per host, that glue hides which device a service may
touch and turns a vendor switch into an edit in every file.

## Principles

- **One namespace, one vendor switch.** `services.infernix.gpu` is consumed by
  the Ollama, llama-swap, and ComfyUI modules and handles
  `HSA_OVERRIDE_GFX_VERSION`, visible device masking, and package selection.
- **Bring your own package set.** Modules take a `pkgs` passthrough, so a
  consumer supplies its `rocmSupport`/`cudaSupport` nixpkgs instantiation instead
  of the flake setting a global flag.
- **Model fetching is pinned and verified.** `modelFetch.jobs` fetch pinned
  Hugging Face artifacts with size/hash verification, ready manifests, and
  exclusive model locks.
- **Fail closed on GPU access.** Colibri serving requires explicit backend
  configuration; manual-lifecycle profiles consume a fresh `admissionMarker`
  and stay loopback-only, out of boot, firewall, and nodectl.
- **Consumer keeps policy.** Model selection, revisions, license acceptance,
  and destination directories are the consumer's decision, not module defaults.

## Scope

- NixOS modules under `nixosModules.default` for GPU, Ollama, llama-swap,
  ComfyUI, model fetch jobs, Colibri, Qdrant, and SurrealDB
- Home Manager modules that translate local Infernix endpoints into Goose
  defaults/providers and yeeHaw steeds
- Workload routing profiles and the `infernix-lb`, `infernix-workerd`, and
  `infernix-workload` crates
- Colibri packaging and revision-pinned profiles

## Direction

Recent work concentrates on coordinating model access: persistent locks that
let publishers and serving children share artifacts, strict-residency
exclusion options for Colibri, workload model defaults for embedding and vision
services, and delegating harness registration to Fleetix rather than owning MCP
integration itself.

## Place in the fleet

Canix consumes Infernix as a flake input, and its Home Manager configuration
enables `services.infernix.*` options for the operator's workstation.
