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
      replace = lib.mkOption {
        type = types.enum [ "guarded" "always" "never" ];
        default = "guarded";
        description = "Replacement policy for an existing context directory.";
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

      replace = lib.mkOption {
        type = types.enum [ "guarded" "always" "never" ];
        default = "guarded";
        description = "Replacement policy for an existing context directory.";
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
