{ fetchzip
, lib
, stdenvNoCC
}:
let
  pin = (builtins.fromJSON (builtins.readFile ../pins.lock.json)).packages.ponytail;
in
stdenvNoCC.mkDerivation {
  pname = "ponytail";
  inherit (pin) version;

  src = fetchzip {
    inherit (pin) url hash;
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
