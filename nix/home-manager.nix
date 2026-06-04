self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.contextualize;
  types = lib.types;
  inherit (lib) mkOption;
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
    export PATH=${lib.escapeShellArg (lib.makeBinPath cfg.runtimePackages)}''${PATH:+:$PATH}
    ${envLoader}
    ${lib.concatMapStringsSep "\n" (envFile: ''
      load_contextualize_env_file ${lib.escapeShellArg envFile}
    '') cfg.envFiles}
    exec ${cfg.package}/bin/contextualize "$@"
  '';
  contextActivationArgs =
    [ "contexts" "hydrate" "--registry" cfg.contexts.registryPath ]
    ++ lib.optional cfg.contexts.activation.strict "--strict";
  contextActivationCommand =
    lib.escapeShellArgs ([ "${wrapper}/bin/contextualize" ] ++ contextActivationArgs);
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
      type = types.package;
      default = self.packages.${pkgs.stdenv.hostPlatform.system}.default;
      description = "Package providing the contextualize command.";
    };

    envFiles = lib.mkOption {
      type = types.listOf types.str;
      default = [];
      description = "Shell env files sourced before running contextualize.";
    };

    runtimePackages = lib.mkOption {
      type = types.listOf types.package;
      default = [ pkgs.git ];
      description = "Packages placed on PATH when running contextualize through the wrapper.";
    };

    enableDirenv = lib.mkOption {
      type = types.bool;
      default = false;
      description = "Manage a local direnv file for contextualize development.";
    };

    devDir = lib.mkOption {
      type = types.str;
      default = "${config.home.homeDirectory}/dev/contextualize";
      description = "Local contextualize checkout used by the direnv file.";
    };

    cxPluginsDevDir = lib.mkOption {
      type = types.nullOr types.str;
      default = "${config.home.homeDirectory}/dev/cx-plugins";
      description = "Optional local cx-plugins checkout used by the direnv file.";
    };

    direnvTarget = lib.mkOption {
      type = types.str;
      default = "dev/contextualize/.envrc";
      description = "Home-relative path for the contextualize direnv file.";
    };

    contexts = {
      enable = mkOption {
        type = types.bool;
        default = false;
        description = "Start registry hydration during Home Manager activation.";
      };

      registryPath = mkOption {
        type = types.str;
        default = "${config.home.homeDirectory}/.config/contextualize/contexts.json";
        description = "Path to the context registry consumed by contextualize contexts hydrate.";
      };

      activation = {
        mode = mkOption {
          type = types.enum [ "service" "inline" ];
          default = if pkgs.stdenv.isLinux then "service" else "inline";
          description = "Run context hydration through a non-blocking user service or inline during activation.";
        };

        after = mkOption {
          type = types.listOf types.str;
          default = [ "writeBoundary" ];
          description = "Home Manager activation steps that must run before context hydration.";
        };

        serviceTimeout = mkOption {
          type = types.str;
          default = "30min";
          description = "Timeout for the context hydration user service.";
        };

        strict = mkOption {
          type = types.bool;
          default = false;
          description = "Fail inline activation or the context hydration service when a context fails.";
        };
      };
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [ wrapper ];

    systemd.user.services.contextualize-contexts-hydrate = lib.mkIf (
      pkgs.stdenv.isLinux
      && cfg.contexts.enable
      && cfg.contexts.activation.mode == "service"
    ) {
      Unit = {
        Description = "Hydrate contextualize context registry";
      };

      Service = {
        Type = "oneshot";
        ExecStart = contextActivationCommand;
        TimeoutStartSec = cfg.contexts.activation.serviceTimeout;
      };
    };

    home.activation.contextualizeDirenv = lib.mkIf cfg.enableDirenv (
      lib.hm.dag.entryAfter [ "writeBoundary" ] ''
        target=${lib.escapeShellArg "${config.home.homeDirectory}/${cfg.direnvTarget}"}
        if [ -L "$target" ]; then
          $DRY_RUN_CMD rm "$target"
        fi
        $DRY_RUN_CMD ${pkgs.coreutils}/bin/install -D -m 0644 ${direnvFile} "$target"
      ''
    );

    home.activation.contextualizeContextsHydrate = lib.mkIf cfg.contexts.enable (
      lib.hm.dag.entryAfter (
        cfg.contexts.activation.after
        ++ lib.optional (cfg.contexts.activation.mode == "service") "reloadSystemd"
      ) ''
        if [ -n "''${DRY_RUN_CMD:-}" ]; then
          echo "contextualize: would hydrate contexts"
        elif [ ${lib.escapeShellArg cfg.contexts.activation.mode} = service ]; then
          ${pkgs.systemd}/bin/systemctl --user start --no-block contextualize-contexts-hydrate.service >/dev/null 2>&1 || true
        else
          ${contextActivationCommand}
        fi
      ''
    );
  };
}
