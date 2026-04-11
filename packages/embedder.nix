{
  writeShellApplication,
  nushell,
  git,
}:
writeShellApplication {
  name = "infernis-embedder";
  runtimeInputs = [nushell git];
  text = ''
    exec ${nushell}/bin/nu ${../indexer/infernis-embedder.nu} "$@"
  '';
  meta.mainProgram = "infernis-embedder";
}
