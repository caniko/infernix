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
  # f creates the inode once. Consumers open it read-only under ProtectSystem.
  anchor = path: "f ${toString path} 0644 root root - -";
}
