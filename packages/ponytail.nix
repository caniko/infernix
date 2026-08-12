{ fetchFromGitHub
, lib
, stdenvNoCC
}:
stdenvNoCC.mkDerivation {
  pname = "ponytail";
  version = "4.8.4-unstable-2026-07-22";

  src = fetchFromGitHub {
    owner = "DietrichGebert";
    repo = "ponytail";
    rev = "16f29800fd2681bdf24f3eb4ccffe38be3baec6b";
    hash = "sha256-Y7d4s7uqjH6IbEXhqAiQ+yaxr6iiGcv2X64LuMtG1T8=";
  };

  dontConfigure = true;
  dontBuild = true;

  installPhase = ''
    mkdir -p "$out"
    cp -a ./. "$out/"
  '';

  meta = {
    description = "Lazy senior developer mode for AI agent harnesses";
    homepage = "https://github.com/DietrichGebert/ponytail";
    license = lib.licenses.mit;
    platforms = lib.platforms.unix;
  };
}
