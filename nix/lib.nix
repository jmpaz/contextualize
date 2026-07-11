{ lib }:

let
  cleanNulls = value:
    if builtins.isAttrs value then
      lib.filterAttrs (_name: item: item != null) (
        lib.mapAttrs (_name: item: cleanNulls item) value
      )
    else if builtins.isList value then
      map cleanNulls value
    else
      value;

  manifestSourceCount = manifest:
    lib.length (lib.filter (value: value != null) [
      manifest.source
      manifest.text
      manifest.data
    ]);

  manifestToRegistry = manifest:
    if manifest.source != null then
      { source = manifest.source; }
    else if manifest.text != null then
      { text = manifest.text; }
    else
      { data = cleanNulls manifest.data; };

  mkRegistry = contexts: {
    version = 1;
    contexts = lib.mapAttrs (_name: context: {
      targetDir = context.targetDir;
      contextDir = context.contextDir;
      replace = context.replace;
      manifest = manifestToRegistry context.manifest;
      origin = context.origin or "nix";
    }) contexts;
  };
in
{
  types = import ./types.nix { inherit lib; };

  inherit cleanNulls manifestSourceCount manifestToRegistry mkRegistry;
}
