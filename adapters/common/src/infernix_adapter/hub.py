"""Hub input download with sha256 verification (interfaces sections 2 and 8)."""

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

from .exits import invalid

TOKEN_FILE_ENV = "INFERNIX_HUB_TOKEN_FILE"
CHUNK = 1 << 20
EXTENSIONS = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "model/gltf-binary": ".glb",
    "application/json": ".json",
    "text/plain": ".txt",
}


def hub_token():
    path = os.environ.get(TOKEN_FILE_ENV)
    if not path:
        raise invalid(f"{TOKEN_FILE_ENV} is not set")
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        raise invalid(f"cannot read hub token file {path}: {error}") from error


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _extension(item):
    suffix = Path(item.role).suffix
    if suffix and len(suffix) <= 6:
        return suffix
    return EXTENSIONS.get(item.mime, ".bin")


def download_inputs(envelope, directory):
    """Fetch every payload input into `directory`, verifying each sha256."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    if not envelope.inputs:
        return []
    if not envelope.hub_url:
        raise invalid("payload.hubUrl is missing but inputs were declared")
    token = hub_token()
    downloaded = []
    for item in envelope.inputs:
        name = f"{item.role}-{item.sha256[:16]}{_extension(item)}"
        path = target / name
        request = urllib.request.Request(
            f"{envelope.hub_url}/v1/assets/{item.asset_id}/content",
            headers={"authorization": f"Bearer {token}"},
        )
        digest = hashlib.sha256()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with open(path, "wb") as handle:
                    for block in iter(lambda: response.read(CHUNK), b""):
                        digest.update(block)
                        handle.write(block)
        except urllib.error.HTTPError as error:
            raise invalid(f"asset {item.asset_id} fetch failed: HTTP {error.code}") from error
        except (urllib.error.URLError, OSError) as error:
            raise invalid(f"asset {item.asset_id} fetch failed: {error}") from error
        actual = digest.hexdigest()
        if actual != item.sha256:
            path.unlink(missing_ok=True)
            raise invalid(f"asset {item.asset_id} sha256 {actual} != {item.sha256}")
        downloaded.append((item, path))
    return downloaded
