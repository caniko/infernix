{
  lib,
  pkgs,
}: {
  command = shared: path: [
    (lib.getExe' pkgs.python3 "python3")
    "${./model-lock.py}"
    (
      if shared
      then "--shared"
      else "--exclusive"
    )
    (toString path)
    "--"
  ];
  # f creates missing parents (root/0755) and the inode once. Consumers open
  # it read-only under ProtectSystem; tmpfiles must never truncate or unlink it.
  anchor = path: "f ${toString path} 0644 root root - -";
}
