"""Missed timers select today's active profile, never an overdue profile."""
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


command = shlex.split(sys.argv[1])
source = Path(command[0]).read_text()
shebang = source.splitlines()[0] + "\n"
switch = re.search(r'exec (\S+) "\$profile"', source).group(1)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    clock = root / "clock"
    clock.write_text(shebang + 'printf "%s\\n" "$FIXTURE_TIME"\n')
    clock.chmod(0o700)
    capture = root / "capture"
    capture.write_text(shebang + 'printf "%s\\n" "$@" > "$FIXTURE_RESULT"\n')
    capture.chmod(0o700)
    script = root / "reconcile"
    script.write_text(source.replace(sys.argv[2], str(clock)).replace(switch, str(capture)))
    script.chmod(0o700)
    result = root / "result"
    for now, expected in (("08:00:00", "night"), ("09:00:00", "day"),
                          ("16:59:59", "day"), ("17:00:00", "night"),
                          ("18:00:00", "night")):
        env = dict(os.environ, FIXTURE_TIME=now, FIXTURE_RESULT=str(result))
        # Multiple persistent timers firing after downtime reconcile identically.
        for _ in range(2):
            subprocess.run([str(script), *command[1:]], env=env, check=True, timeout=10)
            assert result.read_text().splitlines() == [expected, "--restart"]
