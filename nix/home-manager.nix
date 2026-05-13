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
  direnvFile = pkgs.writeText "contextualize.envrc" ''
    ${if cfg.cxPluginsDevDir == null then ''
      use flake ${lib.escapeShellArg cfg.devDir}
    '' else ''
      if [ -d ${lib.escapeShellArg cfg.cxPluginsDevDir} ]; then
        use flake ${lib.escapeShellArg cfg.devDir} --override-input cx-plugins ${lib.escapeShellArg "path:${cfg.cxPluginsDevDir}"}
      else
        use flake ${lib.escapeShellArg cfg.devDir}
      fi
    ''}
    ${envLoader}
    ${lib.concatMapStringsSep "\n" (envFile: ''
      load_contextualize_env_file ${lib.escapeShellArg envFile}
    '') cfg.envFiles}
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

    cxPluginsDevDir = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "${config.home.homeDirectory}/dev/cx-plugins";
      description = "Optional local cx-plugins checkout used by the managed direnv file.";
    };

    direnvTarget = lib.mkOption {
      type = lib.types.str;
      default = "dev/contextualize/.envrc";
      description = "Home-relative path for the managed contextualize direnv file.";
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ wrapper ];

    home.activation.contextualizeDirenv = lib.mkIf cfg.enableDirenv (
      lib.hm.dag.entryAfter [ "writeBoundary" ] ''
        target=${lib.escapeShellArg "${config.home.homeDirectory}/${cfg.direnvTarget}"}
        if [ -L "$target" ]; then
          $DRY_RUN_CMD rm "$target"
        fi
        $DRY_RUN_CMD ${pkgs.coreutils}/bin/install -D -m 0644 ${direnvFile} "$target"
      ''
    );
  };
}
