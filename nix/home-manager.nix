self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.contextualize;
  wrapper = pkgs.writeShellScriptBin "contextualize" ''
    ${lib.concatMapStringsSep "\n" (envFile: ''
      if [ -r ${lib.escapeShellArg envFile} ]; then
        set -a
        . ${lib.escapeShellArg envFile}
        set +a
      fi
    '') cfg.envFiles}
    exec ${cfg.package}/bin/contextualize "$@"
  '';
in
{
  options.programs.contextualize = {
    enable = lib.mkEnableOption "contextualize CLI";

    package = lib.mkOption {
      type = lib.types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      description = "Package providing the contextualize command.";
    };

    envFiles = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      description = "Shell env files sourced before running contextualize.";
    };

    enableDirenv = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Manage a local direnv file for contextualize development.";
    };

    devDir = lib.mkOption {
      type = lib.types.str;
      default = "${config.home.homeDirectory}/dev/contextualize";
      description = "Local contextualize checkout used by the managed direnv file.";
    };

    direnvTarget = lib.mkOption {
      type = lib.types.str;
      default = "dev/contextualize/.envrc";
      description = "Home-relative path for the managed contextualize direnv file.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ wrapper ];

    home.file = lib.mkIf cfg.enableDirenv {
      "${cfg.direnvTarget}".text = ''
        use flake ${cfg.devDir}
        ${lib.concatMapStringsSep "\n" (envFile: ''
          if [ -r ${lib.escapeShellArg envFile} ]; then
            set -a
            . ${lib.escapeShellArg envFile}
            set +a
          fi
        '') cfg.envFiles}
      '';
    };
  };
}
