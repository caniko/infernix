"""Adapter entrypoint: envelope in on stdin, one result object out on stdout."""

import platform
import random
import sys
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

from . import cache, gpu, hub, result, telemetry
from .cancel import CancelWatcher
from .envelope import parse
from .exits import (
    AdapterError,
    ERROR,
    OOM,
)
from .fingerprint import content_fingerprint
from .result import collect
from .threads import bound_threads


@dataclass
class Context:
    """Everything an adapter operation may touch."""

    envelope: object
    inputs: list
    staging_directory: Path
    output_directory: Path
    cancel: CancelWatcher
    fingerprint: str = ""
    validation: dict = field(default_factory=lambda: {"state": "ok", "findings": []})


    @property
    def params(self):
        return self.envelope.params

    @property
    def profile(self):
        return self.envelope.profile

    @property
    def seed(self):
        return self.envelope.seed

    @property
    def operation(self):
        return self.envelope.operation

    def inputs_for(self, role):
        found = [(item, path) for item, path in self.inputs if item.role == role]
        if not found:
            raise AdapterError(14, f"no input with role {role!r}")
        return found

    def input_path(self, role):
        return self.inputs_for(role)[0][1]

    def check_cancel(self):
        self.cancel.check()

    def wait(self, seconds):
        self.cancel.wait(seconds)

    def artifact(self, source, relative, mime):
        """Declare one output file, relative to the output directory."""
        return (source, relative, mime)

    def output(self, relative):
        return self.output_directory / relative


def seed_everything(seed, torch_module=None):
    """The fingerprint covers the seed, so every source of randomness must honour it."""
    value = int(seed)
    random.seed(value)
    try:
        import numpy

        numpy.random.seed(value % (2**32))
    except ImportError:
        pass
    if torch_module is None:
        return value
    torch_module.manual_seed(value)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.manual_seed_all(value)
    return value


def _runtime_versions(package_version, torch_module):
    version = None
    if torch_module is not None:
        version = getattr(torch_module, "__version__", None)
    return {
        "python": platform.python_version(),
        "torch": version,
        "package": package_version,
    }


def run(
    handler,
    *,
    package_version,
    torch_module=None,
    use_cache=True,
    torch_guard=True,
    stdin=None,
):
    """Run one adapter operation and return the process exit code."""
    stdin = stdin or sys.stdin.buffer
    staging = None
    watcher = None
    sampler = None
    try:
        envelope = parse(stdin.read())
        staging = Path(envelope.staging_directory)
        output = Path(envelope.output_directory)
        output.mkdir(parents=True, exist_ok=True)
        bundle = staging / "inputs"
        watcher = CancelWatcher(envelope.cancel_path).start()
        watcher.check()
        bound_threads(envelope.cpu_threads, torch_module)
        requires_gpu = envelope.profile.get("cpuPolicy") != "authoring"
        torch = None
        if requires_gpu and torch_guard:
            torch = gpu.assert_gpu(envelope.profile, torch_module)
        elif requires_gpu:
            torch = torch_module
        runtime = _runtime_versions(package_version, torch or torch_module)

        fingerprint = content_fingerprint(
            profile=envelope.profile,
            runtime=runtime,
            operation=envelope.operation,
            inputs=[item.sha256 for item in envelope.inputs],
            params=envelope.params,
            seed=envelope.seed,
            rig=envelope.params.get("rig"),
            style=envelope.params.get("style"),
        )

        if use_cache:
            root = cache.cache_root()
            manifest = cache.lookup(root, fingerprint, output)
            if manifest is not None:
                watcher.check()
                artifacts = result.from_manifest(manifest, output.name)
                document = result.build_result(
                    artifacts,
                    fingerprint,
                    True,
                    _telemetry(envelope, sampler=None),
                    {"state": "ok", "findings": []},
                )
                result.emit(document)
                return 0

        seed_everything(envelope.seed, torch)
        downloaded = hub.download_inputs(envelope, bundle)
        context = Context(envelope, downloaded, staging, output, watcher, fingerprint)

        with ExitStack() as stack:
            if requires_gpu:
                stack.enter_context(
                    gpu.device_gate(
                        envelope.profile,
                        timeout=_gate_timeout(envelope.params),
                    )
                )
                gpu.ensure_device(envelope.peak_vram_gib, envelope.backend)
                sampler = stack.enter_context(_sampler(envelope.profile))
            specs = handler(context)
            watcher.check()

        artifacts = collect(staging, output, specs)
        if use_cache:
            cache.store(cache.cache_root(), fingerprint, artifacts)
        watcher.check()
        document = result.build_result(
            [item for item in (_artifact_record(a) for a in artifacts)],
            fingerprint,
            False,
            _telemetry(envelope, sampler),
            context.validation,
        )
        result.emit(document)
        return 0
    except AdapterError as error:
        _report(error)
        return error.code
    except BaseException as error:  # noqa: BLE001 - mapped to a typed exit
        if error.__class__.__name__ in ("SystemExit", "KeyboardInterrupt"):
            raise
        if isinstance(error, GeneratorExit):
            raise
        if _is_oom(error):
            typed = AdapterError(OOM, f"out of memory: {error}")
            _report(typed)
            return OOM
        traceback.print_exc()
        _report(AdapterError(ERROR, f"{type(error).__name__}: {error}"))
        return ERROR
    finally:
        if watcher is not None:
            watcher.stop()
        if sampler is not None:
            sampler.stop()


def _artifact_record(artifact):
    return {
        "source": artifact.source,
        "path": artifact.path,
        "sha256": artifact.sha256,
        "mime": artifact.mime,
        "size": artifact.size,
    }


def _is_oom(error):
    from .exits import is_oom

    return is_oom(error)


def _gate_timeout(params):
    value = (params or {}).get("gateTimeoutSeconds")
    return None if value is None else float(value)


class _sampler:
    def __init__(self, profile):
        self.profile = profile
        self.sampler = None

    def __enter__(self):
        self.sampler = telemetry.Sampler(self.profile.get("backend")).start()
        return self.sampler

    def __exit__(self, *exc):
        if self.sampler is not None:
            self.sampler.stop()
        return False


def _telemetry(envelope, sampler):
    summary = sampler.summary() if sampler is not None else {}
    return {
        "peakVramBytes": summary.get("peakVramBytes", 0),
        "gpuSeconds": summary.get("gpuSeconds", 0.0),
        "cpuSeconds": telemetry.cpu_seconds(),
        "peakRssBytes": telemetry.peak_rss_bytes(),
        "maxTempC": summary.get("maxTempC", 0),
        "maxPowerW": summary.get("maxPowerW", 0),
        "device": envelope.device or "",
        "backend": envelope.backend or "",
    }


def _report(error):
    print(str(error), file=sys.stderr)
    for finding in error.findings:
        print(f"  {finding}", file=sys.stderr)
