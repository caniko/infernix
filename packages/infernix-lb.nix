{
  lib,
  rustPlatform,
}:

rustPlatform.buildRustPackage {
  pname = "infernix-lb";
  version = "0.1.0";

  src = lib.cleanSourceWith {
    src = ../.;
    filter = path: type:
      let
        base = baseNameOf path;
      in
        !(type == "directory" && base == "target");
  };

  cargoLock.lockFile = ../Cargo.lock;
  cargoBuildFlags = [
    "-p"
    "infernix-lb"
  ];
  cargoTestFlags = [
    "-p"
    "infernix-lb"
  ];
}
