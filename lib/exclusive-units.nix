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
  # Fail-closed: an unqueryable unit (D-Bus down, unknown name) refuses
  # the start. Only `inactive` and `failed` (no live process) permit it;
  # `activating` refuses, closing the concurrent-start race to the
  # residual window where both starters check before either activates.
  mkExclusiveCondition = pkgs: units:
    if units == [] then [] else
      let
        guard = pkgs.writeShellApplication {
          name = "infernix-exclusive-guard";
          runtimeInputs = [pkgs.systemd];
          text = ''
            set -euo pipefail
            for unit in "$@"; do
              state="$(systemctl show -p ActiveState --value "$unit" 2>/dev/null)" || {
                echo "exclusive-gpu: cannot query $unit, refusing start" >&2
                exit 1
              }
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
