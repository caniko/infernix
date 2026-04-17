{
  description = "infernis — Declarative NixOS modules for AI/ML model serving";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    # Bleeding-edge source for ollama, llama-cpp, and llama-swap only.
    # Everything else (qdrant, curl, nushell, the embedder package, the
    # consumer's own pkgs) stays on nixos-unstable above. See README for
    # the cache-cost rationale behind this split.
    nixpkgs-bleeding.url = "github:NixOS/nixpkgs/master";

    # Local development bridge to Mnemo until this integration is upstreamed.
    mnemo.url = "path:/data/nvme0/can/Projects/mnemo";

    # embr: code embedding indexer (replaces the old nushell indexer).
    # Lives in its own repo so it can be used standalone.
    embr = {
      url = "git+ssh://git@codeberg.org/caniko/rs-embr.git";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    nixpkgs-bleeding,
    mnemo,
    embr,
  }: let
    systems = ["x86_64-linux" "aarch64-linux"];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    nixosModules = {
      default = {
        lib,
        pkgs,
        ...
      }: {
        imports = [
          ./modules/nixos
          # Re-export embr's NixOS module under the same default import
          # path so consumers get `services.embr.*` for free.
          embr.nixosModules.default
        ];
        # Thread the bleeding-edge nixpkgs flake into the module tree so
        # ollama / llama-cpp / llama-swap can re-instantiate it with the
        # consumer's own system + config (GPU flags, allowUnfree, etc.).
        _module.args.infernisBleedingNixpkgs = nixpkgs-bleeding;
        _module.args.infernisMnemo = mnemo;
        # Default `services.embr.package` to the one locked by infernis,
        # picking the binary for the active host system. mkDefault keeps
        # it overridable downstream.
        services.embr.package =
          lib.mkDefault embr.packages.${pkgs.stdenv.hostPlatform.system}.embr;
      };

      mnemo = {
        ...
      }: {
        imports = [./modules/nixos/mnemo.nix];
        _module.args.infernisMnemo = mnemo;
      };
    };

    homeModules = {
      default = import ./modules/home-manager;
      # Opt-in sub-module that writes programs.goose.* from the
      # services.infernis.goose outputs. Only import for users that also
      # import goose-hm's HM module.
      goose = import ./modules/home-manager/goose-programs.nix;
      # Opt-in sub-module that writes programs.yh.steeds from the
      # services.infernis.yeehaw outputs. Only import for users that also
      # import yeeHaw's HM module.
      yeehaw = import ./modules/home-manager/yeehaw-programs.nix;
    };

    packages = forAllSystems (system: {
      embr = embr.packages.${system}.embr;
      mnemo = mnemo.packages.${system}.default;
      default = embr.packages.${system}.embr;
    });
  };
}
