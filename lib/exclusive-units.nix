{lib}: let
  # Each edge gets one stable inode in both directions. Unrelated pairs do
  # not serialize each other, and a unit with several peers holds every edge.
  pairLocks = unit: peers:
    map (peer: "/run/lock/infernix/gpu-${builtins.hashString "sha256" (builtins.toJSON (lib.sort builtins.lessThan [unit peer]))}.lock")
    (lib.unique peers);
in {
  inherit pairLocks;
  anchors = unit: peers:
    lib.optionals (peers != []) (["d /run/lock/infernix 0755 root root - -"]
      ++ map (path: "f ${path} 0644 root root - -") (pairLocks unit peers));
  command = pkgs: unit: peers:
    lib.optionals (peers != []) ([
        (lib.getExe' pkgs.python3 "python3")
        "${./gpu-admission.py}"
      ]
      ++ lib.concatMap (path: ["--lock" path]) (pairLocks unit peers)
      ++ ["--"]);

  # Systemd ExecCondition lines refusing start while any listed unit is
  # anything but inactive/failed. Both directions of an exclusive pair
  # must declare each other: nodectl starts units sequentially, so the
  # second starter refuses instead of co-running, and manual starts get
  # the same refusal. Nothing is ever stopped or killed. This check gives
  # an inactive refusal for an already-visible peer; command's atomic
  # process-lifetime locks close the check-and-start race (EX_CONFIG=78,
  # excluded from automatic restarts by the serving unit).
  #
  # Fail-closed: an unqueryable unit (D-Bus down) refuses the start, and
  # so does any unit that is not loaded (not-found, masked, ...): a
  # nonexistent unit still reports ActiveState=inactive, so an ActiveState
  # check alone would let a typo or renamed peer silently disable
  # exclusion. Only `inactive` and `failed` (no live process) permit it;
  # `activating` refuses as well. The state check alone is not admission.
  # Named properties (no --value): --value order is not contractual.
  mkExclusiveCondition = pkgs: units:
    if units == []
    then []
    else let
      guard = pkgs.writeShellApplication {
        name = "infernix-exclusive-guard";
        runtimeInputs = [pkgs.systemd];
        text = ''
          set -euo pipefail
          for unit in "$@"; do
            props="$(systemctl show -p LoadState,ActiveState "$unit" 2>/dev/null)" || {
              echo "exclusive-gpu: cannot query $unit, refusing start" >&2
              exit 1
            }
            load="" state=""
            while IFS="=" read -r name value; do
              case "$name" in
                LoadState) load="$value" ;;
                ActiveState) state="$value" ;;
              esac
            done <<<"$props"
            case "$load" in
              loaded) ;;
              *)
                echo "exclusive-gpu: refusing start, $unit load state is ''${load:-unknown}" >&2
                exit 1
                ;;
            esac
            case "$state" in
              inactive|failed) ;;
              *)
                echo "exclusive-gpu: refusing start, $unit is $state" >&2
                exit 1
                ;;
            esac
          done
        '';
      };
    in ["${guard}/bin/infernix-exclusive-guard ${lib.escapeShellArgs units}"];
}
