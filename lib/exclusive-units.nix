{ lib }:

{
  # Systemd ExecCondition lines refusing start while any listed unit is
  # anything but inactive/failed. Both directions of an exclusive pair
  # must declare each other: nodectl starts units sequentially, so the
  # second starter refuses instead of co-running, and manual starts get
  # the same refusal. Nothing is ever stopped or killed -- refusal leaves
  # the unit inactive without failing the job, so switch-to-configuration
  # and nodectl resume stay green.
  #
  # Fail-closed: an unqueryable unit (D-Bus down) refuses the start, and
  # so does any unit that is not loaded (not-found, masked, ...): a
  # nonexistent unit still reports ActiveState=inactive, so an ActiveState
  # check alone would let a typo or renamed peer silently disable
  # exclusion. Only `inactive` and `failed` (no live process) permit it;
  # `activating` refuses, closing the concurrent-start race to the
  # residual window where both starters check before either activates.
  # Named properties (no --value): --value order is not contractual.
  mkExclusiveCondition = pkgs: units:
    if units == [] then [] else
      let
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
      in
      ["${guard}/bin/infernix-exclusive-guard ${lib.escapeShellArgs units}"];
}
