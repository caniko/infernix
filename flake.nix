{
  description = "infernis — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Bleeding-edge source for ollama, llama-cpp, and llama-swap only.
    # Everything else (qdrant, curl, nushell, the embedder package, the
    # consumer's own pkgs) stays on nixos-unstable above. See README for
    # the cache-cost rationale behind this split.
    nixpkgs-bleeding.url = "github:NixOS/nixpkgs/master";
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-bleeding,
  }: let
    systems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    nixosModules = {
      default = {
        imports = [./modules/nixos];
        # Thread the bleeding-edge nixpkgs flake into the module tree so
        # ollama / llama-cpp / llama-swap can re-instantiate it with the
        # consumer's own system + config (GPU flags, allowUnfree, etc.).
        _module.args.infernisBleedingNixpkgs = nixpkgs-bleeding;
      };
    };

    homeModules = {
      default = import ./modules/home-manager;
      # Opt-in sub-module that writes programs.goose.* from the
      # services.infernis.goose outputs. Only import for users that also
      # import goose-hm's HM module.
      goose = import ./modules/home-manager/goose-programs.nix;
    };

    packages = forAllSystems (system: let
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      infernis-embedder = pkgs.callPackage ./packages/embedder.nix {};
      default = self.packages.${system}.infernis-embedder;
    });
  };
}
