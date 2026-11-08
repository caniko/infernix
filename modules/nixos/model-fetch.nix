# Pinned, manifest-gated model fetch jobs (`packages/model-fetch`).
#
# Each job stages one pinned revision (repo + rev + exact file list with byte
# sizes), verifies it, and atomically publishes a revision-specific ready
# manifest. Units are manual-only (never WantedBy): the operator runs
# `systemctl start model-fetch-<name>` explicitly, so downloads are never a
# boot, evaluation, or ordinary build dependency, and nothing outside the Nix
# store is touched until the operator acts.
{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib) mkOption types;
  modelFetch = pkgs.callPackage ../../packages/model-fetch {};
  jobType = types.submodule ({config, ...}: {
    options = {
      description = mkOption {
        type = types.str;
        description = "Human-readable purpose of this fetch job.";
      };
      repo = mkOption {
        type = types.str;
        description = "Hugging Face repository holding the snapshot.";
      };
      rev = mkOption {
        type = types.str;
        description = "Pinned commit SHA of the snapshot.";
      };
      files = mkOption {
        type = types.listOf (types.submodule {
          options = {
            name = mkOption {
              type = types.str;
              description = "Repo-relative file path.";
            };
            sizeBytes = mkOption {
              type = types.nullOr types.ints.positive;
              default = null;
              description = "Exact expected byte size, when verified out of band. Null skips the size check (HF metadata verification still applies); the landed size is recorded in the manifest.";
            };
            sha256 = mkOption {
              type = types.nullOr (types.strMatching "[0-9a-f]{64}");
              default = null;
              description = "Pinned sha256 of the file. When set, a download with any other digest fails and a manifest recording another digest re-fetches.";
            };
          };
        });
        default = [];
        description = "Exact files to fetch and verify. Empty fetches the whole revision and enumerates what landed (diagnostic profiles only).";
      };
      totalBytes = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "Declared payload size for space budgeting and ±10% landed-total sanity. Null budgets from the sum of declared file sizes.";
      };
      kind = mkOption {
        type = types.enum ["gguf-file" "comfyui-model-file"];
        description = "Publish layout: per-file move into a shared directory. `comfyui-model-file` keeps each file's repo-relative ComfyUI model folder (clip_vision/, vae/, ...).";
      };
      stagingDir = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Staging directory. Null derives <destDir>/.staging-<job> automatically. Override only to relocate within the same filesystem for atomic publish.";
      };
      destDir = mkOption {
        type = types.path;
        description = "Shared destination directory for GGUF files.";
      };
      manifestName = mkOption {
        type = types.str;
        default = "ready.json";
        description = "Manifest filename inside the destination.";
      };
      lockPath = mkOption {
        type = types.str;
        default = "${toString config.destDir}.doty-lock";
        description = "Persistent lock inode shared by model producers, consumers and retention tooling such as Doty.";
      };
      reserveBytes = mkOption {
        type = types.ints.positive;
        default = 10 * 1024 * 1024 * 1024;
        description = "Free-space reserve that must remain after the payload lands.";
      };
    };
  });
  cfg = config.services.infernix.modelFetch;
  jobJson = name: job: let
    staging =
      if job.stagingDir != null
      then toString job.stagingDir
      else "${toString job.destDir}/.staging-${name}";
  in
    pkgs.writeText "model-fetch-${name}.json" (builtins.toJSON ({
        inherit (job) repo rev kind reserveBytes;
        files = map (f: {inherit (f) name sizeBytes;} // lib.optionalAttrs (f.sha256 != null) {inherit (f) sha256;}) job.files;
        stagingDir = staging;
      }
      // lib.optionalAttrs (job.totalBytes != null) {inherit (job) totalBytes;}
      // {
        publish = {
          type = "file-move";
          destDir = toString job.destDir;
          inherit (job) manifestName;
        };
      }));
in {
  options.services.infernix.modelFetch.jobs = mkOption {
    type = types.attrsOf jobType;
    default = {};
    description = "Pinned, manifest-gated model fetch jobs (manual oneshot units).";
  };

  config = {
    systemd.services = lib.mapAttrs' (name: job:
      lib.nameValuePair "model-fetch-${name}" {
        description = "Fetch pinned model snapshot (${job.description})";
        after = ["network-online.target"];
        wants = ["network-online.target"];
        serviceConfig = {
          Type = "oneshot";
          ExecStart = "${modelFetch}/bin/model-fetch ${jobJson name job}";
          # Multi-GB downloads: no start timeout (failure modes are the
          # script's own loud errors, not the watchdog).
          TimeoutStartSec = "infinity";
          NoNewPrivileges = true;
          PrivateTmp = true;
          UMask = "0022";
        };
      })
    cfg.jobs;
  };
}
