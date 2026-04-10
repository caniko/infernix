{
  description = "infernis — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {self, ...}: {
    nixosModules = {
      default = import ./modules/nixos;
    };

    homeManagerModules = {
      default = import ./modules/home-manager;
    };
  };
}
