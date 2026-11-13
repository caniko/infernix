{
  lib,
  pkgs,
}: rec {
  # One lock file per GPU device, shared between the ledger adapters, which
  # hold it exclusively for a whole job, and the other GPU tenants, which hold
  # it shared for their process lifetime.
  lockPath = device: "/run/infernix/gpu/${device}.lock";

  # Created once, never truncated or unlinked: the inode is the lock.
  tmpfilesRule = device: "f ${lockPath device} 0660 root render -";

  # `flock <file> <command>` execs the command and keeps the descriptor, so the
  # lease spans the command's whole process tree. Default off keeps every
  # existing host byte-identical.
  wrap = {
    enable,
    device,
    command,
  }:
    if !enable
    then command
    else "${lib.getExe' pkgs.util-linux "flock"} --shared ${lib.escapeShellArg (lockPath device)} ${command}";
}
