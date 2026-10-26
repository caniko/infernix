import json, os
def snapshot_download(repo_id=None, revision=None, local_dir=None, allow_patterns=None):
    calls_path = os.environ["STUB_CALLS"]
    with open(calls_path, "w") as handle:
        json.dump({"repo_id": repo_id, "revision": revision,
                   "allow_patterns": allow_patterns}, handle)
    specs = json.loads(os.environ.get("STUB_FILES", "{}"))
    for name, size in specs.items():
        if allow_patterns is not None and name not in allow_patterns:
            continue
        dest = os.path.join(local_dir, name)
        os.makedirs(os.path.dirname(dest) or local_dir, exist_ok=True)
        with open(dest, "wb") as handle:
            handle.write(b"\0" * size)
    return local_dir
