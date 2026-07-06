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
        contextualizeNix = import ./nix/lib.nix { inherit lib; };
        python = pkgs.python312;
        mkContextualize = { contextualizeSrc ? self.outPath, cxPluginsSrc ? cx-plugins, extraPluginSrcs ? [], sourcePreference ? "wheel" }:
          let
            cleanContextualizeSrc = lib.cleanSourceWith {
              src = contextualizeSrc;
              filter = path: type:
                let
                  rel = lib.removePrefix "${toString contextualizeSrc}/" (toString path);
                in
                !(rel == "build"
                  || lib.hasPrefix "build/" rel
                  || rel == "dist"
                  || lib.hasPrefix "dist/" rel
                  || lib.hasPrefix "src/contextualize.egg-info" rel
                  || lib.hasSuffix "/__pycache__" rel
                  || lib.hasInfix "/__pycache__/" rel);
            };
            workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = cleanContextualizeSrc; };
            cxPluginsWorkspace = uv2nix.lib.workspace.loadWorkspace {
              workspaceRoot = cxPluginsSrc;
            };
            extraPluginWorkspaces = map (src:
              uv2nix.lib.workspace.loadWorkspace {
                workspaceRoot = src;
              }
            ) extraPluginSrcs;
            overlay = workspace.mkPyprojectOverlay {
              inherit sourcePreference;
            };
            cxPluginsOverlay = cxPluginsWorkspace.mkPyprojectOverlay {
              inherit sourcePreference;
            };
            extraPluginOverlays = map (pluginWorkspace:
              pluginWorkspace.mkPyprojectOverlay {
                inherit sourcePreference;
              }
            ) extraPluginWorkspaces;
            extraPluginDeps = lib.foldl' (
              deps: pluginWorkspace: deps // pluginWorkspace.deps.default
            ) {} extraPluginWorkspaces;
            pyprojectOverrides = final: prev: {
              contextualize = prev.contextualize.overrideAttrs (_old: {
                src = cleanContextualizeSrc;
              });
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
              pyjwt = prev.pyjwt.overrideAttrs (old: {
                passthru = (old.passthru or {}) // {
                  optional-dependencies = (old.passthru.optional-dependencies or {}) // {
                    crypto = {
                      cryptography = [];
                    };
                  };
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
                lib.composeManyExtensions ([
                  pyproject-build-systems.overlays.wheel
                  overlay
                  cxPluginsOverlay
                ] ++ extraPluginOverlays ++ [
                  pyprojectOverrides
                ])
              );
            unwrapped =
              pythonSet.mkVirtualEnv "contextualize-env-unwrapped" (
                workspace.deps.all // extraPluginDeps
              );
          in
          pkgs.symlinkJoin {
            name = "contextualize-env";
            paths = [ unwrapped ];
            nativeBuildInputs = [ pkgs.makeWrapper ];
            postBuild = ''
              wrapProgram $out/bin/contextualize \
                --prefix PATH : ${lib.makeBinPath [ unwrapped pkgs.deno pkgs.ffmpeg pkgs.poppler-utils ]}
              ensure_completion_dir() {
                if [ -L "$1" ]; then
                  rm "$1"
                fi
                mkdir -p "$1"
              }
              ensure_completion_dir $out/share/bash-completion
              ensure_completion_dir $out/share/bash-completion/completions
              ensure_completion_dir $out/share/zsh
              ensure_completion_dir $out/share/zsh/site-functions
              ensure_completion_dir $out/share/fish
              ensure_completion_dir $out/share/fish/vendor_completions.d
              _CONTEXTUALIZE_COMPLETE=bash_source $out/bin/contextualize > $out/share/bash-completion/completions/contextualize
              _CONTEXTUALIZE_COMPLETE=zsh_source $out/bin/contextualize > $out/share/zsh/site-functions/_contextualize
              _CONTEXTUALIZE_COMPLETE=fish_source $out/bin/contextualize > $out/share/fish/vendor_completions.d/contextualize.fish
            '';
          };
        venv = mkContextualize {};
        pluginCheck = pkgs.runCommand "contextualize-plugin-check" { } ''
          ${venv}/bin/python - <<'PY'
          import importlib.metadata as metadata

          import pylatexenc
          import selectolax
          import yt_dlp
          from contextualize.plugins.loader import get_loaded_plugins
          from cx_plugins.providers.ytdlp import plugin as ytdlp_plugin

          names = {plugin.name for plugin in get_loaded_plugins()}
          assert "ytdlp" in names
          assert metadata.version("yt-dlp")
          assert ytdlp_plugin.can_resolve("https://youtu.be/vjqt8T3tJIE", {})
          PY
          ${venv}/bin/contextualize plugins > plugins.txt
          grep -q '^-' plugins.txt
          grep -q 'ytdlp' plugins.txt
          touch $out
        '';
      in
      {
        packages.default = venv;
        packages.contextualize = venv;

        apps.default = {
          type = "app";
          program = "${venv}/bin/contextualize";
        };

        checks.default = venv;
        checks.plugins = pluginCheck;

        lib = {
          mkContextualize = mkContextualize;
          nix = contextualizeNix;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            venv
            pkgs.deno
            pkgs.exiftool
            pkgs.ffmpeg
            pkgs.git
            pkgs.poppler-utils
            pkgs.python312
            pkgs.uv
          ];
          env = {
            UV_PYTHON = python.interpreter;
            UV_PYTHON_DOWNLOADS = "never";
          };
          shellHook = ''
            contextualize_dev_root=$(git -C . rev-parse --show-toplevel 2>/dev/null || pwd)
            contextualize_pythonpath="$contextualize_dev_root/src"
            if [ -d "$contextualize_dev_root/../cx-plugins/src" ]; then
              contextualize_pythonpath="$contextualize_pythonpath:$contextualize_dev_root/../cx-plugins/src"
            fi
            export PYTHONPATH="$contextualize_pythonpath"
          '';
        };
      }) // {
        homeManagerModules.default = import ./nix/home-manager.nix self;
      };
}
