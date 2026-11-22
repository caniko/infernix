"""File-backed BF16 weights and synchronous, single-block GPU transfers."""

import gc
import hashlib
import json
import math
import mmap
import os
import resource
import stat
from contextlib import contextmanager
from pathlib import Path
from time import monotonic

from infernix_adapter.exits import invalid

GIB = 1 << 30
SIZE = (1024, 1024)
SIGMAS = (1, 0.9, 0.75, 0.5, 0)
CHECKPOINT_MARKER = {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}
LIGHTNING_TARGETS = (
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
    "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj", "txt_mlp.net.2",
)


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise invalid("JSON contains duplicate keys")
    return result


def readonly(path, kind):
    if not kind(path.lstat().st_mode) or os.access(path, os.W_OK):
        raise invalid("model trees must contain only read-only regular files and directories")


def walk_error(error):
    raise error


def verify_models(pins, locations):
    paths = {}
    for role, pin in pins.items():
        root = locations[role]
        if not root.is_absolute() or root.name != pin["rev"] or root.resolve(strict=True) != root:
            raise invalid(f"{role} must be a canonical, revision-addressed local directory")
        readonly(root, stat.S_ISDIR)
        manifest = root / "ready.json"
        readonly(manifest, stat.S_ISREG)
        if manifest.stat().st_size > 64 << 10:
            raise invalid(f"{role} ready manifest is too large")
        actual = json.loads(manifest.read_bytes(), object_pairs_hook=unique_object)
        expected = sorted(pin["files"], key=lambda f: f["name"])
        if (
            not isinstance(actual, dict)
            or type(actual.get("schemaVersion")) is not int or actual["schemaVersion"] != 1
            or actual.get("repo") != pin["repo"]
            or actual.get("rev") != pin["rev"]
            or not isinstance(actual.get("files"), list)
            or any(not isinstance(f, dict) or not isinstance(f.get("name"), str) for f in actual["files"])
            or sorted(actual["files"], key=lambda f: f["name"]) != expected
            or actual.get("totalBytes") != sum(f["sizeBytes"] for f in expected)
        ):
            raise invalid(f"{role} ready manifest differs from the pinned inventory")
        if role == "snapshot":
            observed = set()
            for directory, folders, files in os.walk(root, onerror=walk_error):
                for folder in folders:
                    readonly(Path(directory) / folder, stat.S_ISDIR)
                observed.update(str((Path(directory) / file).relative_to(root)) for file in files)
            if observed != {"ready.json", *(f["name"] for f in expected)}:
                raise invalid("official snapshot must contain exactly the pinned loader files")
        for entry in expected:
            path = root / entry["name"]
            for parent in path.relative_to(root).parents:
                readonly(root / parent, stat.S_ISDIR)
            readonly(path, stat.S_ISREG)
            if path.stat().st_size != entry["sizeBytes"]:
                raise invalid(f"{role} file size differs from its pin")
            # Provisioning verifies every SHA before publishing the trusted ready
            # manifest. Recheck configs and LoRA; do not pre-read 55 GiB of weights.
            if path.suffix != ".safetensors" or role == "lightning":
                with path.open("rb") as handle:
                    if hashlib.file_digest(handle, "sha256").hexdigest() != entry["sha256"]:
                        raise invalid(f"{role} file digest differs from its pin")
        paths[role] = root if role == "snapshot" else path
    return paths


def bf16_ranges(header, length):
    ranges = []
    for entry in header.values():
        if not isinstance(entry, dict):
            raise invalid("invalid BF16 tensor metadata")
        shape, offsets = entry.get("shape"), entry.get("data_offsets")
        if (
            entry.get("dtype") != "BF16"
            or not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape)
            or not isinstance(offsets, list) or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            raise invalid("invalid BF16 tensor metadata")
        start, end = offsets
        if not 0 <= start < end <= length or end - start != 2 * math.prod(shape):
            raise invalid("invalid BF16 tensor size or range")
        ranges.append((start, end))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise invalid("BF16 tensor ranges overlap or leave a gap")
        cursor = end
    if not ranges or cursor != length:
        raise invalid("BF16 tensor ranges do not cover the checkpoint payload")


def assign_denoiser(torch, kind, snapshot, handle):
    count = int.from_bytes(handle.read(8), "little")
    if not 0 < count <= 1 << 20 or count % 2:
        raise invalid("invalid denoiser safetensors header length")
    header = json.loads(handle.read(count), object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise invalid("invalid denoiser safetensors header")
    header.pop("__metadata__", None)
    if header.pop("__index_timestep_zero__", None) != CHECKPOINT_MARKER:
        raise invalid("denoiser is missing the Qwen 2511 conditioning marker")
    cfg = kind.load_config(snapshot, subfolder="transformer", local_files_only=True)
    if cfg["zero_cond_t"] is not True:
        raise invalid("denoiser requires the Qwen 2511 zero conditioning timestep")
    with torch.device("meta"):
        model = kind.from_config(cfg)
    rope = model.pos_embed
    model.pos_embed = type(rope)(theta=rope.theta, axes_dim=rope.axes_dim, scale_rope=rope.scale_rope)
    expected = model.state_dict()
    if len(model.transformer_blocks) != 60 or header.keys() != expected.keys():
        raise invalid("denoiser keys do not match the pinned Qwen 2511 model")
    # ACCESS_COPY is a private view of the original file, not a second checkpoint.
    mapping = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
    bf16_ranges(header, len(mapping) - count - 8)
    state = {}
    for key, entry in header.items():
        a, b = entry["data_offsets"]
        if tuple(entry["shape"]) != tuple(expected[key].shape):
            raise invalid("denoiser tensor shape does not match the pinned config")
        state[key] = torch.frombuffer(
            mapping, dtype=torch.bfloat16, count=(b - a) // 2, offset=8 + count + a,
        ).view(entry["shape"])
    model.load_state_dict(state, assign=True, strict=True)
    return model, mapping


def install_lightning(torch, model, state):
    from torch.nn.functional import linear

    names = [f"transformer_blocks.{i}.{suffix}" for i in range(60) for suffix in LIGHTNING_TARGETS]
    parts = (".lora_down.weight", ".lora_up.weight", ".alpha")
    if set(state) != {name + part for name in names for part in parts}:
        raise invalid("Lightning keys do not match the 720 denoiser-only targets")
    for name in names:
        layer = model.get_submodule(name)
        a, b, alpha = (state[name + part] for part in parts)
        if (
            not isinstance(layer, torch.nn.Linear)
            or any(t.dtype != torch.bfloat16 for t in (a, b, alpha))
            or a.shape != (64, layer.in_features) or b.shape != (layer.out_features, 64)
            or alpha.ndim != 0 or not math.isfinite(scale := alpha.item() / 64) or scale <= 0
        ):
            raise invalid("Lightning expects BF16 rank-64 linears and finite positive alpha")
        layer.register_buffer("lightning_a", a, persistent=False)
        layer.register_buffer("lightning_b", b, persistent=False)
        layer.register_forward_hook(
            lambda m, args, out, scale=scale: out + linear(linear(args[0], m.lightning_a), m.lightning_b) * scale
        )


def stage(sources, device):
    for obj, original in sources.items():
        # Direct DMA from mapped storage dirtied ROCm's host pages in the pilot.
        obj.data = original.clone().to(device, non_blocking=False)


def restore(sources):
    for obj, original in sources.items():
        obj.data = original  # never copy back from the GPU


@contextmanager
def file_backed_groups(torch, model, device, reclaim):
    from diffusers.hooks.group_offloading import ModuleGroup

    def initialize(group):
        objects = [*group.parameters, *group.buffers]
        for module in group.modules:
            objects.extend(module.parameters())
            objects.extend(module.buffers())
        return {obj: obj.detach() for obj in objects}

    def onload(group):
        stage(group.cpu_param_dict, device)
        reclaim()

    def offload(group):
        restore(group.cpu_param_dict)

    # Diffusers 0.43.0's private hooks are isolated to this one-render process.
    saved = (ModuleGroup._init_cpu_param_dict, ModuleGroup._onload_from_memory,
             ModuleGroup._offload_to_memory)
    ModuleGroup._init_cpu_param_dict = initialize
    ModuleGroup._onload_from_memory = onload
    ModuleGroup._offload_to_memory = offload
    try:
        model.enable_group_offload(
            onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, low_cpu_mem_usage=True, use_stream=False,
        )
        yield
    finally:
        (ModuleGroup._init_cpu_param_dict, ModuleGroup._onload_from_memory,
         ModuleGroup._offload_to_memory) = saved


def load_encoder(torch, snapshot, device, backend):
    from safetensors.torch import load_file
    from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    directory = snapshot / "text_encoder"
    index = json.loads((directory / "model.safetensors.index.json").read_bytes())["weight_map"]
    state = {}
    shards = sorted(directory.glob("*.safetensors"))
    if {p.name for p in shards} != set(index.values()):
        raise invalid("encoder shard index does not cover the pinned files")
    for file in shards:
        shard = load_file(file)
        if set(shard) != {key for key, name in index.items() if name == file.name}:
            raise invalid("encoder shard keys differ from the pinned index")
        if any(t.dtype != torch.bfloat16 for t in shard.values()):
            raise invalid("encoder weights must be BF16")
        state.update(shard)
    del shard
    enc = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        None, config=Qwen2_5_VLConfig.from_pretrained(directory, local_files_only=True),
        state_dict=state, dtype=torch.bfloat16, local_files_only=True,
    )
    del state
    if enc.dtype != torch.bfloat16 or any(p.is_meta for p in enc.parameters()):
        raise invalid("encoder parameters are not fully assigned BF16 weights")
    def move(tensor):
        return tensor.clone().to(device, non_blocking=False)

    if backend == "rocm":
        return enc._apply(move)
    # The complete encoder peaks above a 16 GiB laptop GPU. Keep its vision
    # tower/embeddings on GPU, stream its text layers at the same BF16 precision.
    layers = enc.model.language_model.layers
    streamed = {id(child) for block in layers for child in block.modules()}
    for module in enc.modules():
        if id(module) not in streamed:
            module._apply(move, recurse=False)
    for block in layers:
        sources = {t: t.detach() for t in (*block.parameters(), *block.buffers())}
        block.register_forward_pre_hook(lambda _m, _args, src=sources: stage(src, device))
        block.register_forward_hook(lambda _m, _args, _out, src=sources: restore(src), always_call=True)
    return enc


def render(torch, device, cg, backend, paths, image, instruction, seed):
    # Must precede the first Hugging Face import in this fresh process.
    os.environ.update(
        HF_HUB_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TRANSFORMERS_OFFLINE="1",
        HF_DEACTIVATE_ASYNC_LOAD="1", HF_HUB_DISABLE_PROGRESS_BARS="1",
    )
    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats(device)
    started = monotonic()
    phase = {}
    steps = []
    advice = 0
    snapshot = paths["snapshot"]
    with paths["denoiser"].open("rb") as handle:
        model, mapping = assign_denoiser(torch, QwenImageTransformer2DModel, snapshot, handle)
        install_lightning(torch, model, load_file(paths["lightning"]))

        def reclaim():
            nonlocal advice
            if int((cg / "memory.current").read_text()) >= 19 * GIB:
                mapping.madvise(mmap.MADV_DONTNEED)
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                advice += 1

        with file_backed_groups(torch, model, device, reclaim):
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                str(snapshot), text_encoder=None, transformer=model, dtype=torch.bfloat16,
                local_files_only=True, use_safetensors=True,
            )
            if pipe.vae.dtype != torch.bfloat16:
                raise invalid("VAE did not load at BF16")
            pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config({
                "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
                "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
                "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
                "time_shift_type": "exponential", "use_beta_sigmas": False, "use_dynamic_shifting": True,
                "use_exponential_sigmas": False, "use_karras_sigmas": False,
            })
            pipe.set_progress_bar_config(disable=True)
            pipe.vae._apply(lambda tensor: tensor.clone().to(device, non_blocking=False))
            phase["pipelineReady"] = monotonic() - started
            pipe.register_modules(text_encoder=load_encoder(torch, snapshot, device, backend))
            phase["encoderReady"] = monotonic() - started
            prepare = pipe.prepare_latents
            step_at = None

            def prepare_latents(*args, **kwargs):
                nonlocal step_at
                pipe.text_encoder = None
                gc.collect()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                phase["encoderReleased"] = monotonic() - started
                result = prepare(*args, **kwargs)
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                step_at = monotonic()
                phase["latentsReady"] = step_at - started
                return result

            def step(pipeline, i, _t, tensors):
                nonlocal step_at
                torch.cuda.synchronize(device)
                now = monotonic()
                sigmas = pipeline.scheduler.sigmas.tolist()
                if len(sigmas) != len(SIGMAS) or not all(
                    math.isclose(a, b, abs_tol=1e-6) for a, b in zip(sigmas, SIGMAS)
                ):
                    raise invalid("sampler did not use the pinned four-update Euler schedule")
                if i != len(steps) or i >= 4 or step_at is None:
                    raise invalid("sampler did not execute the expected update sequence")
                steps.append(round(now - step_at, 3))
                step_at = now
                return tensors

            pipe.prepare_latents = prepare_latents
            with torch.cuda.device(device), torch.inference_mode():
                output = pipe(
                    image=[image], prompt=instruction, true_cfg_scale=1, negative_prompt=None,
                    width=SIZE[0], height=SIZE[1], num_inference_steps=4,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    callback_on_step_end=step, callback_on_step_end_tensor_inputs=[],
                ).images[0]
            torch.cuda.synchronize(device)
            if len(steps) != 4 or output.size != SIZE or any(p.device.type != "cpu" for p in model.parameters()):
                raise invalid("incomplete render or denoiser weights not released")
            phase["decoded"] = monotonic() - started
    memory = {
        "cgroupPeakBytes": int((cg / "memory.peak").read_text()),
        "cgroupEvents": {k: int(v) for k, v in (line.split() for line in (cg / "memory.events").read_text().splitlines())},
        "processPeakRssBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "allocatorPeakReservedBytes": torch.cuda.max_memory_reserved(device),
        "allocatorPeakAllocatedBytes": torch.cuda.max_memory_allocated(device),
        "denoiserReclaims": advice,
    }
    return output, {"phaseElapsedSeconds": {k: round(v, 3) for k, v in phase.items()},
                    "stepSeconds": steps}, memory
