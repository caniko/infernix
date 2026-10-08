"""Run rendered activations and prove excluded harness trees stay byte-identical."""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def snapshot(home, selected):
    result = {}
    for path in home.rglob("*"):
        relative = path.relative_to(home)
        if selected and (relative == selected or selected in relative.parents):
            continue
        metadata = path.lstat()
        contents = (os.readlink(path) if path.is_symlink()
                    else path.read_bytes() if path.is_file() else None)
        result[str(relative)] = (stat.S_IMODE(metadata.st_mode), contents)
    return result


for harness, activation in zip(("codex", "opencode", "empty"), sys.argv[1:], strict=True):
    with tempfile.TemporaryDirectory() as root:
        root = Path(root)
        home = root / "home"
        home.mkdir()
        sentinels = {
            "AGENTS.md": "user guidance\n",
            ".agents/rules/ponytail.md": "user agents rules\n",
            ".codex/config.toml": "[hooks.state]\n",
            ".codex/hooks.json": '{"hooks":{}}\n',
            ".claude/settings.json": '{"hooks":{}}\n',
            ".claude/CLAUDE.md": "user Claude guidance\n",
            ".config/opencode/opencode.json": '{"plugin":["keep-me"]}\n',
            ".config/opencode/AGENTS.md": "user OpenCode guidance\n",
            ".copilot/copilot-instructions.md": "user Copilot guidance\n",
            ".config/amp/AGENTS.md": "user Amp guidance\n",
            ".config/swival/AGENTS.md": "user Swival guidance\n",
            ".cursor/rules/ponytail.mdc": "user Cursor guidance\n",
            ".windsurf/rules/ponytail.md": "user Windsurf guidance\n",
            ".clinerules/ponytail.md": "user Cline guidance\n",
            ".kiro/steering/ponytail.md": "user Kiro guidance\n",
            ".qoder/rules/ponytail.md": "user Qoder guidance\n",
        }
        for relative, contents in sentinels.items():
            path = home / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        hermes_log = root / "hermes.log"
        hermes = bin_dir / "hermes"
        hermes.write_text('#!/bin/sh\necho invoked >> "$HERMES_LOG"\n')
        hermes.chmod(0o755)
        selected = {"codex": Path(".codex"), "opencode": Path(".config/opencode")}.get(harness)
        before = snapshot(home, selected)
        env = dict(os.environ, HOME=str(home), HERMES_LOG=str(hermes_log),
                   PATH=f"{bin_dir}:{os.environ['PATH']}")
        for _ in range(2):
            subprocess.run([activation], env=env, check=True, timeout=60)
            assert snapshot(home, selected) == before, f"{harness} modified an excluded harness"
            assert not hermes_log.exists(), f"{harness} invoked excluded Hermes"
        if harness == "codex":
            assert (home / ".codex/plugins/ponytail").is_symlink()
            assert "[marketplaces.ponytail]" in (home / ".codex/config.toml").read_text()
            hooks = json.loads((home / ".codex/hooks.json").read_text())
            assert len(hooks["hooks"]["SessionStart"]) == 1
        elif harness == "opencode":
            plugins = json.loads((home / ".config/opencode/opencode.json").read_text())["plugin"]
            assert len(plugins) == 2 and "keep-me" in plugins
        else:
            assert not Path("/tmp/infernix-ponytail-selection-empty").exists()

print("codex-only, opencode-only and empty selections preserve excluded harnesses")
