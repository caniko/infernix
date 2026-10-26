{
  lib,
  pkgs,
}:
pkgs.stdenvNoCC.mkDerivation {
  pname = "infernix-codex-provider";
  version = "0.1.0";
  src = ../codex-provider;
  dontConfigure = true;
  dontBuild = true;
  nativeBuildInputs = [pkgs.makeWrapper];

  doCheck = true;
  checkPhase = "${pkgs.nodejs}/bin/node server.mjs --self-test";

  installPhase = ''
    mkdir -p $out/libexec/infernix-codex-provider $out/bin
    cp server.mjs $out/libexec/infernix-codex-provider/server.mjs
    makeWrapper ${pkgs.nodejs}/bin/node $out/bin/infernix-codex-provider \
      --add-flags "$out/libexec/infernix-codex-provider/server.mjs"
  '';

  meta = {
    description = "Loopback OpenAI-compatible provider backed by the Codex CLI";
    homepage = "https://infernix.tartanoglu.com";
    license = lib.licenses.mit;
    mainProgram = "infernix-codex-provider";
    platforms = lib.platforms.unix;
  };
}
