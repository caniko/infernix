{
  description = "infernis — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {...}: {
    nixosModules = {
      default = import ./modules/nixos;
    };

    homeModules = {
      default = import ./modules/home-manager;
      # Opt-in sub-module that writes programs.goose.* from the
      # services.infernis.goose outputs. Only import for users that also
      # import goose-hm's HM module.
      goose = import ./modules/home-manager/goose-programs.nix;
    };
  };
}
