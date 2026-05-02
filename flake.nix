{
  description = "Model context preparation utility";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
    };
    cx-plugins = {
      url = "github:jmpaz/cx-plugins";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, cx-plugins, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = nixpkgs.lib;
        python = pkgs.python312;
        mkContextualize = { cxPluginsSrc ? cx-plugins, sourcePreference ? "wheel" }:
          let
            workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
            overlay = workspace.mkPyprojectOverlay {
              inherit sourcePreference;
            };
            pyprojectOverrides = final: prev: {
              "cx-plugins" = prev."cx-plugins".overrideAttrs (_old: {
                src = cxPluginsSrc;
              });
              grapheme = prev.grapheme.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {
                  setuptools = [];
                };
              });
              pylatexenc = prev.pylatexenc.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {
                  setuptools = [];
                };
              });
              pyperclip = prev.pyperclip.overrideAttrs (old: {
                nativeBuildInputs = (old.nativeBuildInputs or []) ++ final.resolveBuildSystem {
                  setuptools = [];
                };
              });
              magika = prev.magika.overrideAttrs (_old: {
                dontAutoPatchelf = true;
              });
              referencing = prev.referencing.overrideAttrs (_old: {
                dontAutoPatchelf = true;
              });
            };
            pythonSet =
              (pkgs.callPackage pyproject-nix.build.packages {
                inherit python;
              }).overrideScope (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.wheel
                  overlay
                  pyprojectOverrides
                ]
              );
          in
          pythonSet.mkVirtualEnv "contextualize-env" workspace.deps.all;
        venv = mkContextualize {};
      in
      {
        packages.default = venv;
        packages.contextualize = venv;

        apps.default = {
          type = "app";
          program = "${venv}/bin/contextualize";
        };

        checks.default = venv;

        lib.mkContextualize = mkContextualize;

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.exiftool
            pkgs.ffmpeg
            pkgs.git
            pkgs.python312
            pkgs.uv
          ];
          env = {
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
        };
      }) // {
        homeManagerModules.default = import ./nix/home-manager.nix self;
      };
}
