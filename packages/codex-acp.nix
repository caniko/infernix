{ lib, pkgs }:
pkgs.buildNpmPackage rec {
  pname = "codex-acp";
  version = "1.1.4";

  src = pkgs.fetchFromGitHub {
    owner = "agentclientprotocol";
    repo = "codex-acp";
    rev = "v${version}";
    hash = "sha256-oBg/i4ewa6dF7d/lK0JaNOCBrgXTdsltLB+xvwXAV7E=";
  };

  npmDepsHash = "sha256-r1c2Z2TbcU0X6mUdF5jpu3ldLnK+Yd+r0qQzjRHJ0mw=";

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
