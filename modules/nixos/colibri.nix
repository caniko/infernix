# Colibri local-inference servers (`coli serve`, OpenAI-compatible).
#
# One systemd unit per enabled profile, plus a rev-pinned manifest-gated
# fetch unit per profile. A profile serves only when its weights ready
# manifest exists AND names the configured revision; the manifest is
# published atomically by the fetch unit, so a fresh host never boot-loops
# on missing weights and never serves partial state.
#
# GPU execution is explicit and fail-closed, never a silent fallback:
# - `backend` selects the build flavor; the serving package MUST carry the
#   same backend in `passthru.colibriBackend` (asserted). A CPU-built
#   package with `backend = "cuda"` fails evaluation instead of serving
#   on CPU while claiming VRAM.
# - Only engines whose MoE expert execution the GPU tier covers may use a
#   non-cpu backend (see `gpuTierEngines`). The GLM engine compiles CUDA
#   paths for resident dense tensors only -- its experts stream from disk
#   -- so a hip/cuda build does NOT make glm-family profiles VRAM
#   inference, and the assertion below rejects that combination rather
#   than letting hybrid RAM inference wear a VRAM label.
#
# Lifecycle WITHOUT health coupling: enabled profiles join nodectl
# drain/resume via the fleet node's `units`, but the module never touches
# `healthUnits`. Consumers MUST pin healthUnits explicitly when enabling
# any profile (asserted); a down Colibri must never withdraw working
# routes from the load balancer by default.
{
  config,
  lib,
  pkgs,
  ...
}: let
  inherit (lib)
    concatStringsSep
    escapeShellArgs
    filterAttrs
    flip
    getExe'
    mapAttrsToList
    mapAttrs'
    mkEnableOption
    mkIf
    mkMerge
    mkOption
    nameValuePair
    optionalAttrs
    optionals
    types
    ;

  cfg = config.services.infernix.colibri;

  colibriPackaging = import ../../lib/colibri-packaging.nix { inherit lib; };

  profileType = types.submodule ({ name, ... }: {
    options = {
      enable = mkEnableOption "Colibri ${name} profile";
      port = mkOption {
        type = types.port;
        description = "TCP port for the OpenAI-compatible API.";
      };
      bind = mkOption {
        type = types.str;
        default = "0.0.0.0";
        description = "Bind address. Upstream fails closed on non-loopback binds without COLI_API_KEY; the key is always set here.";
      };
      modelDir = mkOption {
        type = types.path;
        description = "Final model directory published atomically by the fetch unit (must contain ready.json).";
      };
      stagingDir = mkOption {
        type = types.path;
        description = "Scratch directory for verified downloads (same filesystem as modelDir parent for atomic rename).";
      };
      modelId = mkOption {
        type = types.str;
        description = "Served model ID advertised on /v1/models.";
      };
      engine = mkOption {
        type = types.enum [
          "glm"
          "glm53"
          "inkling"
          "kimi"
          "olmoe"
          "qwen36"
          "qwen38"
          "deepseek_v4"
          "deepseek_v41"
        ];
        description = "Upstream engine family serving this snapshot.";
      };
      backend = mkOption {
        type = types.enum colibriPackaging.backends;
        default = "cpu";
        description = "Execution backend. Anything but cpu requires a matching package build (asserted) and a tier-covered engine (asserted). There is no vulkan here: enabling it is a separate validation decision, never a silent fallback.";
      };
      gpuDevices = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "0";
        description = "COLI_GPUS device selection (e.g. \"0\" or \"0,1\"). Null leaves the engine default.";
      };
      expertGb = mkOption {
        type = types.nullOr types.numbers.positive;
        default = null;
        description = "CUDA_EXPERT_GB: VRAM budget for the expert tier. Null leaves the engine default.";
      };
      releaseHost = mkOption {
        type = types.bool;
        default = false;
        description = "CUDA_RELEASE_HOST=1: GPU-tier experts drop host backing after upload (VRAM as additional pinned capacity at zero RAM cost; the engine rematerializes from disk on CPU-path misses).";
      };
      ctxSize = mkOption {
        type = types.ints.positive;
        description = "Server context window (--ctx).";
      };
      ngen = mkOption {
        type = types.ints.positive;
        description = "Default per-request generation cap (--ngen); harnesses may request less.";
      };
      kvSlots = mkOption {
        type = types.addCheck types.int (slots: slots >= 1 && slots <= 16);
        default = 1;
        description = "Concurrent generation slots (upstream range 1-16; families clamp further at launch, loudly).";
      };
      expertSlotsPerLayer = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        example = 256;
        description = ''
          Expert cache slots per layer (--cap). The qwen36 VRAM tier
          activates ONLY when this equals the model's expert count
          (cap == n_experts): anything less loads a partial CPU cache and
          the engine disables the tier loudly
          ([qtier] cap=N != n_experts=M -> tier disabled), leaving pure
          CPU inference that VRAM readings alone would misdiagnose as a
          slow tier. Null omits --cap and takes upstream's legacy default
          (8 for non-GLM engines: tier always disabled). Required for
          non-cpu backends (asserted): a gpu backend without an explicit
          expert count is a misconfiguration, never a default.
        '';
      };
      maxQueue = mkOption {
        type = types.ints.unsigned;
        default = 4;
        description = "Bounded admission queue (upstream default 8); small because each turn is disk-bound.";
      };
      queueTimeout = mkOption {
        type = types.numbers.positive;
        default = 100;
        description = "Queue wait budget in seconds; stays under the 120s gateway per-target timeout.";
      };
      memoryMaxGib = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "systemd MemoryMax in GiB; null leaves the host default. Always paired with MemorySwapMax=0. This caps RAM it may use -- it must never be raised to evade a VRAM requirement.";
      };
      apiKeyFile = mkOption {
        type = types.path;
        description = "Raw COLI_API_KEY secret path; loaded as a systemd credential, never on argv.";
      };
      admissionMarker = mkOption {
        type = types.nullOr types.str;
        default = null;
        example = "/var/lib/infernix-colibri-kat-coder/admission-approved";
        description = ''
          Operator-created flag file the serve unit additionally requires
          (ANDed with the weights manifest condition). wantedBy=[] does not
          stop switch-to-configuration from starting new units, so this is
          the actual manual-start gate: absent marker skips the unit without
          failing boot or switch. Create it only after the GPU-admission
          protocol verifies the GPU is drained for this profile; remove it
          to revoke.
        '';
      };
      hfTokenPath = mkOption {
        type = types.nullOr types.path;
        default = null;
        description = "Optional HuggingFace token file for gated repos (Authorization header on fetch).";
      };
      weightsRepo = mkOption {
        type = types.str;
        description = "Hugging Face repo holding the snapshot.";
      };
      weightsRev = mkOption {
        type = types.str;
        description = "Pinned commit SHA of the snapshot.";
      };
      weightsFiles = mkOption {
        type = types.listOf (types.submodule {
          options = {
            name = mkOption {
              type = types.str;
              description = "Repo-relative file path.";
            };
            sizeBytes = mkOption {
              type = types.nullOr types.ints.positive;
              default = null;
              description = "Exact expected byte size, when verified out of band.";
            };
          };
        });
        default = [ ];
        description = "Files to fetch (empty = whole revision).";
      };
      weightsTotalBytes = mkOption {
        type = types.nullOr types.ints.positive;
        default = null;
        description = "Declared payload size for budgeting and landed-total sanity (±10%).";
      };
      reserveBytes = mkOption {
        type = types.ints.unsigned;
        default = 10 * 1024 * 1024 * 1024;
        description = "Free-space reserve that must remain after the payload lands.";
      };
      blockedReason = mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "When set, enabling this profile fails evaluation with the reason. Unblocks by editing data, not by flag.";
      };
      heavyweight = mkOption {
        type = types.bool;
        default = true;
        description = "Heavyweight profiles mutually exclude each other (disk/RAM streaming pressure).";
      };
    };
  });

  enabledProfiles = filterAttrs (_: profile: profile.enable) cfg.profiles;

  entrypoint = pkgs.writeTextFile {
    name = "infernix-colibri-entrypoint.py";
    executable = true;
    destination = "/bin/infernix-colibri-entrypoint";
    # Launch gate: the weights ready-manifest must name the configured
    # rev+repo and list exactly the configured files (multiset), every file
    # must be present at its recorded size, and the API key must arrive via
    # the systemd credential directory -- never argv. Sizes establish
    # presence and length only; full hashes were verified at fetch time.
    text = ''
      #!${getExe' pkgs.python3 "python3"}
      import json
      import os
      import sys


      def fail(message):
          raise SystemExit("infernix-colibri-entrypoint: %s" % message)


      def main():
          if len(sys.argv) != 2:
              fail("usage: infernix-colibri-entrypoint <config.json>")
          with open(sys.argv[1], encoding="utf-8") as handle:
              config = json.load(handle)

          model_dir = config["modelDir"]
          manifest_path = os.path.join(model_dir, "ready.json")
          manifest = None
          try:
              with open(manifest_path, encoding="utf-8") as handle:
                  manifest = json.load(handle)
          except FileNotFoundError:
              fail("missing ready manifest %s; provision weights first "
                   "(systemctl start %s)" % (manifest_path, config["fetchUnit"]))
          except json.JSONDecodeError as error:
              fail("corrupt ready manifest %s: %s" % (manifest_path, error))
          assert manifest is not None  # fail() above always raises
          if manifest.get("rev") != config["rev"]:
              fail("weights rev mismatch: manifest has %r, service wants %r "
                   "(re-provision %s)" % (manifest.get("rev"), config["rev"], model_dir))
          if config.get("repo") is not None and manifest.get("repo") != config["repo"]:
              fail("weights repo mismatch: manifest has %r, service wants %r "
                   "(re-provision %s)" % (manifest.get("repo"), config["repo"], model_dir))
          expected = config.get("files")
          if not isinstance(expected, list) or not expected:
              fail("service config lists no expected files; refusing to serve an "
                   "unverifiable directory")
          for name in expected:
              if not isinstance(name, str) or not name or name.startswith("/") or ".." in name.split("/"):
                  fail("unsafe expected file name %r in service config" % (name,))
          if len(set(expected)) != len(expected):
              fail("duplicate file names in service config files list")
          listed = manifest.get("files")
          if not isinstance(listed, list) or not listed:
              fail("weights manifest lists no files; refusing to serve an "
                   "unverifiable directory (re-provision %s)" % model_dir)
          listed_names = []
          for entry in listed:
              if not isinstance(entry, dict):
                  fail("malformed manifest entry %r (re-provision %s)" % (entry, model_dir))
              name = entry.get("name")
              if not isinstance(name, str) or not name or name.startswith("/") or ".." in name.split("/"):
                  fail("unsafe manifest file name %r (re-provision %s)" % (name, model_dir))
              listed_names.append(name)
              size = entry.get("sizeBytes")
              if type(size) is not int or size < 0:
                  fail("manifest entry %r records no valid size; refusing to serve "
                       "unverifiable weights (re-provision %s)" % (name, model_dir))
          if sorted(listed_names) != sorted(expected):
              fail("weights inventory mismatch: manifest lists %d files, service "
                   "expects %d (re-provision %s)"
                   % (len(listed_names), len(expected), model_dir))
          for entry in listed:
              path = os.path.join(model_dir, entry["name"])
              if not os.path.isfile(path):
                  fail("manifest-listed weights file missing: %s "
                       "(re-provision %s)" % (path, model_dir))
              if os.path.getsize(path) != entry["sizeBytes"]:
                  fail("weights size drift for %s (re-provision %s)" % (path, model_dir))

          credential_dir = os.environ.get("CREDENTIALS_DIRECTORY", "")
          key_path = os.path.join(credential_dir, "coli-api-key")
          api_key = ""
          try:
              with open(key_path, encoding="utf-8") as handle:
                  api_key = handle.read().strip()
          except FileNotFoundError:
              fail("missing API key credential %s" % key_path)
          if not api_key:
              fail("empty API key credential %s" % key_path)

          argv = [
              "coli", "serve",
              "--model", model_dir,
              "--host", config["bind"],
              "--port", str(config["port"]),
              "--model-id", config["modelId"],
              "--ctx", str(config["ctxSize"]),
              "--ngen", str(config["ngen"]),
              "--max-queue", str(config["maxQueue"]),
              "--queue-timeout", str(config["queueTimeout"]),
              "--kv-slots", str(config["kvSlots"]),
          ]
          if config.get("expertSlotsPerLayer") is not None:
              argv += ["--cap", str(config["expertSlotsPerLayer"])]
          env = dict(os.environ)
          env["COLI_API_KEY"] = api_key
          os.execve(config["coliBin"], argv, env)


      if __name__ == "__main__":
          main()
    '';
  };

  fetchTool = pkgs.writeShellApplication {
    name = "infernix-colibri-fetch";
    runtimeInputs = with pkgs; [curl jq coreutils gnugrep gawk];
    # Rev-pinned, manifest-gated fetch. The job (argv[1], JSON) carries
    # repo/rev/exact files/staging/publish/disk reserve. Readiness is the
    # FULL artifact identity (rev, repo, names, sizes); a manifest for the
    # same rev but a different file set, a deleted file, or size drift all
    # re-fetch instead of skipping. Refusal never deletes: a missing shard
    # beside a matching manifest fails closed WITHOUT touching the
    # installed snapshot, and an existing finalDir without a matching
    # manifest is never clobbered.
    text = ''
      set -euo pipefail
      job="$1"
      repo=$(jq -r '.repo' "$job")
      rev=$(jq -r '.rev' "$job")
      staging=$(jq -r '.stagingDir' "$job")
      final=$(jq -r '.publish.finalDir' "$job")
      reserve=$(jq -r '.reserveBytes' "$job")
      budget=$(jq -r '.totalBytes // 0' "$job")
      token_path=$(jq -r '.hfTokenPath // empty' "$job")
      manifest="$final/ready.json"

      auth=()
      if [ -n "$token_path" ]; then
        auth=(-H "Authorization: Bearer $(cat "$token_path")")
      fi

      file_count=$(jq '.files | length' "$job")
      if [ -f "$manifest" ] \
        && [ "$(jq -r '.rev' "$manifest")" = "$rev" ] \
        && [ "$(jq -r '.repo' "$manifest")" = "$repo" ] \
        && [ "$(jq -c '[.files[].name] | sort' "$manifest")" = "$(jq -c '[.files[].name] | sort' "$job")" ]; then
        ok=1
        while IFS= read -r name; do
          want=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes' "$job")
          have_size=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes' "$manifest")
          if [ ! -f "$final/$name" ] || [ "$(stat -c%s "$final/$name")" != "$have_size" ]; then
            ok=0
            break
          fi
          if [ "$want" != "null" ] && [ "$want" != "$have_size" ]; then
            ok=0
            break
          fi
        done < <(jq -r '.files[].name' "$job")
        if [ "$ok" = 1 ]; then
          echo "infernix-colibri-fetch: already provisioned $repo @''${rev:0:12}"
          exit 0
        fi
        echo "infernix-colibri-fetch: $final is incomplete for the recorded manifest; refusing to delete the installed snapshot" >&2
        exit 1
      fi
      if [ -e "$final" ]; then
        echo "infernix-colibri-fetch: $final exists without a matching manifest; refusing to clobber unknown state" >&2
        exit 1
      fi

      if [ "$file_count" -eq 0 ]; then
        echo "infernix-colibri-fetch: whole-revision fetch is not supported; declare weightsFiles" >&2
        exit 1
      fi
      declared=$(jq '[.files[].sizeBytes // 0] | add' "$job")
      if [ "$declared" -eq 0 ] && [ "$budget" -eq 0 ]; then
        echo "infernix-colibri-fetch: job declares files but no sizes and no totalBytes" >&2
        exit 1
      fi
      need=$budget
      if [ "$need" -eq 0 ]; then need=$declared; fi
      free=$(df --output=avail -B1 "$(dirname "$staging")" | tail -1 | tr -d ' ')
      if [ "$free" -lt $((need + reserve)) ]; then
        echo "infernix-colibri-fetch: insufficient space: need $need payload + $reserve reserve, have $free free" >&2
        exit 1
      fi

      rm -rf "$staging"
      mkdir -p "$staging"
      while IFS= read -r name; do
        want=$(jq -r --arg n "$name" '.files[] | select(.name == $n) | .sizeBytes // empty' "$job")
        echo "infernix-colibri-fetch: downloading $name"
        curl --fail --show-error --location --retry 3 \
          "''${auth[@]}" \
          "https://huggingface.co/$repo/resolve/$rev/$name" \
          -o "$staging/$name"
        actual=$(stat -c%s "$staging/$name")
        if [ -n "$want" ] && [ "$actual" != "$want" ]; then
          echo "infernix-colibri-fetch: size mismatch for $name: declared $want, disk has $actual" >&2
          exit 1
        fi
      done < <(jq -r '.files[].name' "$job")

      # Record identity + hashes, then publish atomically: the manifest is
      # always the last object to appear inside the renamed directory.
      entries_tmp="$staging/.entries.jsonl"
      : > "$entries_tmp"
      while IFS= read -r name; do
        size=$(stat -c%s "$staging/$name")
        hash=$(sha256sum "$staging/$name" | awk '{print $1}')
        jq -cn --arg n "$name" --argjson s "$size" --arg h "$hash" \
          '{name: $n, sizeBytes: $s, sha256: $h}' >> "$entries_tmp"
      done < <(jq -r '.files[].name' "$job")
      total=$(jq -s '[.[].sizeBytes] | add' "$entries_tmp")
      if [ "$budget" -ne 0 ]; then
        low=$((budget * 9 / 10))
        high=$((budget * 11 / 10))
        if [ "$total" -lt "$low" ] || [ "$total" -gt "$high" ]; then
          echo "infernix-colibri-fetch: total $total outside ±10% of declared $budget" >&2
          exit 1
        fi
      fi
      jq -cn --arg repo "$repo" --arg rev "$rev" --argjson total "$total" \
        --slurpfile files "$entries_tmp" \
        '{schemaVersion: 1, kind: "colibri-dir", repo: $repo, rev: $rev,
          files: $files, totalBytes: $total,
          completedAtUtc: (now | todate)}' > "$staging/ready.json"
      mkdir -p "$(dirname "$final")"
      mv "$staging" "$final"
      echo "infernix-colibri-fetch: published $repo @''${rev:0:12} ($total bytes)"
    '';
  };

  serveConfig = name: profile:
    pkgs.writeText "infernix-colibri-${name}.json" (builtins.toJSON {
      coliBin = "${cfg.package}/bin/coli";
      profile = name;
      modelDir = toString profile.modelDir;
      inherit (profile) bind;
      inherit (profile) port;
      inherit (profile) modelId;
      inherit (profile) ctxSize;
      inherit (profile) ngen;
      inherit (profile) maxQueue;
      inherit (profile) queueTimeout;
      inherit (profile) kvSlots;
      expertSlotsPerLayer = profile.expertSlotsPerLayer;
      rev = profile.weightsRev;
      repo = profile.weightsRepo;
      files = map (f: f.name) profile.weightsFiles;
      fetchUnit = "infernix-colibri-fetch-${name}.service";
    });

  fetchJob = name: profile:
    pkgs.writeText "infernix-colibri-fetch-${name}.json" (builtins.toJSON {
      repo = profile.weightsRepo;
      rev = profile.weightsRev;
      files = map (f: ({name = f.name;} // optionalAttrs (f.sizeBytes != null) {sizeBytes = f.sizeBytes;})) profile.weightsFiles;
      totalBytes = profile.weightsTotalBytes;
      stagingDir = toString profile.stagingDir;
      reserveBytes = profile.reserveBytes;
      hfTokenPath = profile.hfTokenPath;
      publish.finalDir = toString profile.modelDir;
    });

  backendEnv = profile:
    optionalAttrs (profile.backend != "cpu") {
      COLI_CUDA = "1";
      CUDA_RELEASE_HOST = if profile.releaseHost then "1" else "0";
    }
    // optionalAttrs (profile.gpuDevices != null) {COLI_GPUS = profile.gpuDevices;}
    // optionalAttrs (profile.expertGb != null) {CUDA_EXPERT_GB = toString profile.expertGb;};

  nodeName = if cfg.nodeName != null then cfg.nodeName else config.services.infernix.fleet.localNodeName;

  globalChecks =
    let
      pkgBackend = cfg.package.passthru.colibriBackend or "cpu";
    in
    [
      {
        message = "services.infernix.colibri: enabling any profile requires services.infernix.colibri.package";
        ok = enabledProfiles == { } || cfg.package != null;
      }
      {
        message = "services.infernix.colibri: every enabled profile's backend must equal the package build flavor (passthru.colibriBackend); a mismatch would silently serve CPU while claiming VRAM";
        ok = cfg.package == null
          || builtins.all (profile: profile.backend == pkgBackend) (builtins.attrValues enabledProfiles);
      }
      {
        message = "services.infernix.colibri: non-cpu backends require expertSlotsPerLayer (the qwen36 VRAM tier activates only at cap == n_experts; without it the engine silently serves CPU)";
        ok = builtins.all
          (profile: profile.backend == "cpu" || profile.expertSlotsPerLayer != null)
          (builtins.attrValues enabledProfiles);
      }
      {
        message = "services.infernix.colibri: only engines with GPU-tier expert execution (${concatStringsSep ", " colibriPackaging.gpuTierEngines}) may use a non-cpu backend -- the GLM engine streams experts from disk, so a gpu build does not make it VRAM inference";
        ok = builtins.all
          (profile: profile.backend == "cpu" || builtins.elem profile.engine colibriPackaging.gpuTierEngines)
          (builtins.attrValues enabledProfiles);
      }
      {
        message = "services.infernix.colibri: at most one heavyweight profile may be enabled (disk/RAM streaming pressure)";
        ok = builtins.length (builtins.attrValues (filterAttrs (_: profile: profile.enable && profile.heavyweight) cfg.profiles)) <= 1;
      }
      {
        message = "services.infernix.colibri: enabling any profile requires pinning the fleet node's healthUnits explicitly -- a down Colibri must never withdraw working routes by default (null healthUnits gates on every lifecycle unit)";
        ok = enabledProfiles == { }
          || (nodeName != null
            && (config.services.infernix.fleet.nodes.${nodeName}.healthUnits or null) != null);
      }
    ];

  profileChecks = name: profile: [
    {
      message = "services.infernix.colibri.profiles.${name}.queueTimeout must stay under the 120s gateway per-target timeout";
      ok = profile.queueTimeout < 120;
    }
    {
      message = "services.infernix.colibri.profiles.${name} is blocked: ${toString profile.blockedReason}";
      ok = !profile.enable || profile.blockedReason == null;
    }
  ];

  allChecks = globalChecks ++ builtins.concatLists (mapAttrsToList profileChecks cfg.profiles);
in
{
  options.services.infernix.colibri = {
    package = mkOption {
      type = types.nullOr types.package;
      default = null;
      description = "Colibri package providing bin/coli. Must carry passthru.colibriBackend matching every enabled profile's backend (asserted: a CPU-built package never serves a gpu-backend profile).";
    };
    evalChecks = mkOption {
      type = types.listOf (types.submodule {
        options = {
          message = mkOption { type = types.str; readOnly = true; };
          ok = mkOption { type = types.bool; readOnly = true; };
        };
      });
      readOnly = true;
      default = allChecks;
      description = "Introspection seam for checks: every fail-closed predicate with its verdict. `assertions` below is derived from this list, so checks can read verdicts without forcing unrelated modules' assertion messages (some of which only render on failure).";
    };
    nodeName = mkOption {
      type = types.nullOr types.str;
      default = null;
      description = "Fleet node receiving the lifecycle units. Defaults to services.infernix.fleet.localNodeName.";
    };
    openFirewallInterfaces = mkOption {
      type = types.listOf types.str;
      default = [ ];
      example = ["wg-home"];
      description = "Network interfaces where enabled profile ports are opened.";
    };
    profiles = mkOption {
      type = types.attrsOf profileType;
      default = { };
      description = "Named serve profiles; at most one heavyweight profile may be enabled.";
    };
  };

  config = mkMerge [
    {
      assertions = map (check: { assertion = check.ok; message = check.message; }) cfg.evalChecks;
    }

    (mkIf (enabledProfiles != { } && nodeName != null) {
      # Lifecycle: enabled profiles join nodectl drain/resume. Health is
      # deliberately untouched here (see the assertion above).
      services.infernix.fleet.nodes.${nodeName}.units =
        mapAttrsToList (name: _: "infernix-colibri-${name}.service") enabledProfiles;

      networking.firewall.interfaces = builtins.listToAttrs (map
        (iface: nameValuePair iface {
          allowedTCPPorts = mapAttrsToList (_: profile: profile.port) enabledProfiles;
        })
        cfg.openFirewallInterfaces);

      systemd.services = mkMerge (flip mapAttrsToList enabledProfiles (name: profile: {
        "infernix-colibri-${name}" = {
          description = "Infernix Colibri inference (${name}, ${profile.modelId})";
          wantedBy = ["multi-user.target"];
          after = ["network-online.target"];
          wants = ["network-online.target"];
          # Missing or foreign-revision weights are an operator step, not a
          # boot failure; the entrypoint re-validates the manifest anyway.
          # wantedBy=[] does NOT stop switch-to-configuration from starting
          # a new unit (observed: it starts every unit new in the
          # generation). admissionMarker is the real manual-start gate: an
          # operator-created flag the unit requires before it may run at
          # boot, at switch, or by hand. Absence skips the unit WITHOUT
          # failing the switch; revoking (rm) re-arms manual control.
          # Acceptance and GPU-admission protocols create the marker only
          # after verifying the GPU is drained for this profile.
          unitConfig.ConditionPathExists = [
            "${profile.modelDir}/ready.json"
          ] ++ lib.optionals (profile.admissionMarker != null) [
            profile.admissionMarker
          ];
          environment = backendEnv profile;
          serviceConfig =
            {
              Type = "exec";
              DynamicUser = true;
              StateDirectory = "infernix-colibri-${name}";
              WorkingDirectory = "/var/lib/infernix-colibri-${name}";
              # A list would render as repeated ExecStart directives, which
              # systemd only allows for Type=oneshot: join into one command.
              ExecStart = escapeShellArgs ["${getExe' pkgs.python3 "python3"}" "${entrypoint}/bin/infernix-colibri-entrypoint" "${serveConfig name profile}"];
              LoadCredential = ["coli-api-key:${profile.apiKeyFile}"];
              Restart = "on-failure";
              RestartSec = "10s";
              TimeoutStartSec = "15min";
              NoNewPrivileges = true;
              PrivateTmp = true;
              ProtectSystem = "strict";
              ProtectHome = true;
              ReadWritePaths = ["/var/lib/infernix-colibri-${name}"];
              MemorySwapMax = 0;
              MemoryZSwapMax = 0;
              UMask = "0077";
            }
            // optionalAttrs (profile.memoryMaxGib != null) {MemoryMax = "${toString profile.memoryMaxGib}G";};
        };
        "infernix-colibri-fetch-${name}" = {
          description = "Infernix Colibri ${name} weights (${profile.weightsRepo})";
          # A list would render as repeated ExecStart directives, which
          # systemd only allows for Type=oneshot: join into one command.
          serviceConfig = {
            Type = "oneshot";
            ExecStart = escapeShellArgs ["${fetchTool}/bin/infernix-colibri-fetch" "${fetchJob name profile}"];
          };
        };
      }));
    })
  ];
}
