{lib, pkgs}: let
  pin = (builtins.fromJSON (builtins.readFile ../pins.lock.json)).packages.codex-acp;
in pkgs.buildNpmPackage {
  pname = "codex-acp";
  inherit (pin) version;

  src = pkgs.fetchzip {
    inherit (pin) url hash;
  };

  npmDepsHash = pin.fod_hash;

  nativeBuildInputs = [pkgs.makeWrapper];

  installPhase = ''
    runHook preInstall
    mkdir -p $out/libexec/codex-acp $out/bin
    cp dist/index.js $out/libexec/codex-acp/index.js
    makeWrapper ${lib.getExe pkgs.nodejs} $out/bin/codex-acp \
      --add-flags "$out/libexec/codex-acp/index.js" \
      --set-default CODEX_PATH "${lib.getExe pkgs.codex}"
    runHook postInstall
  '';

  meta = {
    description = "Official Agent Client Protocol adapter for Codex";
    homepage = "https://github.com/agentclientprotocol/codex-acp";
    license = lib.licenses.asl20;
    mainProgram = "codex-acp";
    platforms = lib.platforms.unix;
  };
}
