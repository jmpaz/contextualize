{ lib }:

let
  types = lib.types;
  manifestSource = types.submodule {
    options = {
      source = lib.mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Path to a contextualize manifest source.";
      };

      text = lib.mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Inline contextualize manifest source text.";
      };

      data = lib.mkOption {
        type = types.nullOr types.anything;
        default = null;
        description = "Already evaluated contextualize manifest data.";
      };
    };
  };

  contextDeclaration = types.submodule {
    options = {
      contextDir = lib.mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Optional hydration root, resolved relative to the repository.";
      };

      replace = lib.mkOption {
        type = types.enum [ "guarded" "always" "never" ];
        default = "guarded";
        description = "Replacement policy for an existing context directory.";
      };

      origin = lib.mkOption {
        type = types.str;
        default = "nix";
        description = "Registry origin label displayed by contextualize.";
      };

      designations = lib.mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Routing designations exposed by the context registry.";
      };

      manifest = lib.mkOption {
        type = manifestSource;
        default = {};
        description = "Manifest source for this context.";
      };
    };
  };

  registryEntry = types.submodule {
    options = {
      targetDir = lib.mkOption {
        type = types.str;
        description = "Repository or project directory hydrated by this context.";
      };

      contextDir = lib.mkOption {
        type = types.nullOr types.str;
        default = null;
        description = "Optional hydration root, resolved relative to targetDir.";
      };

      replace = lib.mkOption {
        type = types.enum [ "guarded" "always" "never" ];
        default = "guarded";
        description = "Replacement policy for an existing context directory.";
      };

      origin = lib.mkOption {
        type = types.str;
        default = "nix";
        description = "Registry origin label displayed by contextualize.";
      };

      designations = lib.mkOption {
        type = types.listOf types.str;
        default = [];
        description = "Routing designations exposed by the context registry.";
      };

      manifest = lib.mkOption {
        type = manifestSource;
        default = {};
        description = "Manifest source for this context.";
      };
    };
  };
in
{
  inherit manifestSource contextDeclaration registryEntry;
}
