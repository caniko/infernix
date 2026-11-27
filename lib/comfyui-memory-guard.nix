# Separate from the engine cgroup: failure or missed heartbeats fence the engine.
{
  pkgs,
  admitGiB,
  admitPsi,
  tripGiB,
  tripPsi,
  emergencyGiB,
}:
pkgs.writeShellScript "comfyui-memory-guard" ''
  set -eu
  export LC_ALL=C
  meminfo=$1 pressure=$2 notify=$3
  shift 3
  sample() {
    ${pkgs.gawk}/bin/awk -v floor="$1" -v maxpsi="$2" -v phase="$3" -v emergency=${toString emergencyGiB} '
      $1 == "MemAvailable:" && $2 ~ /^[0-9]+$/ && $3 == "kB" { available = $2; mem++ }
      $1 == "full" && $2 ~ /^avg10=[0-9]+\.[0-9]+$/ { psi = substr($2, 7) + 0; full++ }
      END {
        if (mem != 1 || full != 1) reason = "missing or invalid host memory/PSI"
        else if (available < emergency * 1024 * 1024) reason = "emergency floor (" emergency " GiB)"
        else if (available < floor * 1024 * 1024) reason = "host reserve (" floor " GiB)"
        else if (psi >= maxpsi) reason = "sustained full memory PSI (avg10 >= " maxpsi "%)"
        else exit 0
        print "ComfyUI " phase ": " reason ", MemAvailable=" available " kB, full avg10=" psi > "/dev/stderr"
        exit 1
      }
    ' "$meminfo" "$pressure"
  }
  sample ${toString admitGiB} ${builtins.toJSON admitPsi} "admission denied"
  if [ "$#" -gt 0 ]; then
    if ! "$@"; then
      echo 'ComfyUI: GPU admission denied' >&2
      exit 1
    fi
    # Measurement can take several seconds. Recheck before signalling readiness.
    sample ${toString admitGiB} ${builtins.toJSON admitPsi} "admission denied"
  fi
  "$notify" --ready
  while true; do
    sample ${toString tripGiB} ${builtins.toJSON tripPsi} "watchdog trip"
    "$notify" WATCHDOG=1
    ${pkgs.coreutils}/bin/sleep 0.25
  done
''
