"""Worker envelope v2 parsing (interfaces section 2)."""

import json
from dataclasses import dataclass, field

from .exits import invalid

PROTOCOL_VERSION = 2
WORKLOAD = "animation"


@dataclass
class Input:
    asset_id: str
    sha256: str
    mime: str
    role: str


@dataclass
class Envelope:
    queue: str
    job_id: str
    lease_token: str
    attempt: int
    input_fingerprint: str
    staging_directory: str
    output_directory: str
    cancel_path: str
    stage: str
    operation: str
    generation_request_id: str
    hub_url: str
    profile: dict
    inputs: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    seed: int = 0

    @property
    def backend(self):
        return self.profile.get("backend")

    @property
    def device(self):
        return self.profile.get("device")

    @property
    def cpu_threads(self):
        return int(self.profile.get("cpuThreads") or 4)

    @property
    def peak_vram_gib(self):
        peak = self.profile.get("peakVramGib")
        return None if peak is None else float(peak)


def _require(obj, key, kind, where):
    value = obj.get(key)
    if not isinstance(value, kind) or (kind is str and not value):
        raise invalid(f"envelope: {where}{key} missing or not {kind.__name__}")
    return value


def parse(raw):
    """Parse one envelope JSON object from bytes."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as error:
        raise invalid(f"envelope: not JSON: {error}") from error
    if not isinstance(doc, dict):
        raise invalid("envelope: not a JSON object")
    version = doc.get("protocolVersion")
    if version != PROTOCOL_VERSION:
        raise invalid(f"envelope: protocolVersion {version!r} is not {PROTOCOL_VERSION}")
    if doc.get("workload") != WORKLOAD:
        raise invalid(f"envelope: workload {doc.get('workload')!r} is not {WORKLOAD}")

    payload = _require(doc, "payload", dict, "")
    profile = _require(payload, "profile", dict, "payload.")
    inputs = payload.get("inputs") or []
    if not isinstance(inputs, list):
        raise invalid("envelope: payload.inputs is not a list")
    parsed_inputs = []
    for index, item in enumerate(inputs):
        if not isinstance(item, dict):
            raise invalid(f"envelope: payload.inputs[{index}] is not an object")
        parsed_inputs.append(
            Input(
                asset_id=_require(item, "assetId", str, f"payload.inputs[{index}]."),
                sha256=_require(item, "sha256", str, f"payload.inputs[{index}]."),
                mime=item.get("mime") or "application/octet-stream",
                role=item.get("role") or "input",
            )
        )

    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise invalid("envelope: payload.params is not an object")
    seed = payload.get("seed", 0)
    if not isinstance(seed, int):
        raise invalid("envelope: payload.seed is not an integer")

    return Envelope(
        queue=_require(doc, "queue", str, ""),
        job_id=_require(doc, "jobId", str, ""),
        lease_token=_require(doc, "leaseToken", str, ""),
        attempt=_require(doc, "attempt", int, ""),
        input_fingerprint=_require(doc, "inputFingerprint", str, ""),
        staging_directory=_require(doc, "stagingDirectory", str, ""),
        output_directory=_require(doc, "outputDirectory", str, ""),
        cancel_path=_require(doc, "cancelPath", str, ""),
        stage=_require(payload, "stage", str, "payload."),
        operation=_require(payload, "operation", str, "payload."),
        generation_request_id=_require(payload, "generationRequestId", str, "payload."),
        hub_url=(payload.get("hubUrl") or "").rstrip("/"),
        profile=profile,
        inputs=parsed_inputs,
        params=params,
        seed=seed,
    )
