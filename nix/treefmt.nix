{harbor-rs, rustfmtPackage}: {pkgs, ...}: {
  imports = [
    harbor-rs.inputs.harbor-meta.treefmtModules.nix
    harbor-rs.inputs.harbor-meta.treefmtModules.toml
    harbor-rs.treefmtModules.rust
  ];
  projectRootFile = "flake.nix";

  # One repository-wide formatter covers both Nix and every Rust member.
  # Use the same pinned nightly toolchain as the project dev shells.
  programs.rustfmt = {
    edition = "2021";
    package = rustfmtPackage;
  };
}
