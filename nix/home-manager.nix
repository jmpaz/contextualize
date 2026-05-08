self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.contextualize;
  envLoader = ''
    load_contextualize_env_file() {
      env_file=$1
      [ -r "$env_file" ] || return 0
      while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
          ""|\#*) continue ;;
          export\ *) line=''${line#export } ;;
        esac
        case "$line" in
          *=*) key=''${line%%=*}; value=''${line#*=} ;;
          *) continue ;;
        esac
        case "$key" in
          ""|[0-9]*|*[!A-Za-z0-9_]* ) continue ;;
          *) export "$key=$value" ;;
        esac
      done < "$env_file"
    }
  '';
  wrapper = pkgs.writeShellScriptBin "contextualize" ''
    ${envLoader}
    ${lib.concatMapStringsSep "\n" (envFile: ''
      load_contextualize_env_file ${lib.escapeShellArg envFile}
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
        ${envLoader}
        ${lib.concatMapStringsSep "\n" (envFile: ''
          load_contextualize_env_file ${lib.escapeShellArg envFile}
        '') cfg.envFiles}
      '';
    };
  };
}
