{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  boost,
  gtest,
}: let
  version = "1.14.0";
in
  stdenv.mkDerivation {
    pname = "rhubarb-lip-sync";
    inherit version;
    # pin-hold: Keep this release paired with the tested build and resource layout below.
    src = fetchFromGitHub {
      owner = "DanielSWolf";
      repo = "rhubarb-lip-sync";
      tag = "v${version}";
      hash = "sha256-w6TgklwHtwzSHsyz4akT6Nqr/mD8IOGppTMXpikMIHo=";
    };
    nativeBuildInputs = [cmake];
    buildInputs = [boost];
    # Upstream fetches googletest at configure time; use nixpkgs' source.
    cmakeFlags = ["-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${gtest.src}"];
    # Skip the extras (Gradle-based editor plugins).
    makeFlags = ["rhubarb"];
    # Upstream locates res/ relative to the real executable (/proc/self/exe),
    # so keep both together in libexec and expose a symlink in bin.
    # ponytail: tests/resources are skipped; add when a regression needs them.
    installPhase = ''
      runHook preInstall
      mkdir -p $out/bin $out/libexec/rhubarb
      cp rhubarb/rhubarb $out/libexec/rhubarb/
      cp -r rhubarb/res $out/libexec/rhubarb/res
      ln -s $out/libexec/rhubarb/rhubarb $out/bin/rhubarb
      runHook postInstall
    '';
    meta = with lib; {
      description = "Lip sync from audio: mouth-shape cues for 2D animation";
      homepage = "https://github.com/DanielSWolf/rhubarb-lip-sync";
      license = licenses.mit;
      mainProgram = "rhubarb";
      platforms = platforms.linux;
      maintainers = [];
    };
  }
