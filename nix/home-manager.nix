self:
{ config, lib, pkgs, ... }:

let
  cfg = config.programs.contextualize;
  types = lib.types;
  inherit (lib) mkOption;
  manifestFreeformType = types.attrsOf types.anything;
  cleanNulls = value:
    if builtins.isAttrs value then
      lib.filterAttrs (_name: item: item != null) (
        lib.mapAttrs (_name: item: cleanNulls item) value
      )
    else if builtins.isList value then
      map cleanNulls value
    else
      value;
  manifestDataType =
    let
      componentType = types.submodule ({ ... }: {
        freeformType = manifestFreeformType;
        options = {
          name = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Component name.";
          };

          group = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Component group name.";
          };

          components = mkOption {
            type = types.nullOr (types.listOf componentType);
            default = null;
            description = "Nested grouped components.";
          };

          files = mkOption {
            type = types.nullOr (types.listOf types.anything);
            default = null;
            description = "Manifest file specifications.";
          };

          repos = mkOption {
            type = types.nullOr (types.listOf types.anything);
            default = null;
            description = "Manifest repository specifications.";
          };

          text = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Inline component text.";
          };

          prefix = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Text prepended to the component payload.";
          };

          suffix = mkOption {
            type = types.nullOr types.str;
            default = null;
            description = "Text appended to the component payload.";
          };

          "strip-paths" = mkOption {
            type = types.nullOr types.bool;
            default = null;
            description = "Whether to strip common source path prefixes.";
          };

          "target-depth" = mkOption {
            type = types.nullOr types.int;
            default = null;
            description = "Embedded target traversal depth.";
          };

          "target-scope" = mkOption {
            type = types.nullOr (types.enum [ "first" "all" ]);
            default = null;
            description = "Embedded target traversal scope.";
          };

          "include-parent" = mkOption {
            type = types.nullOr types.bool;
            default = null;
            description = "Whether to include the parent target during traversal.";
          };

          gitignore = mkOption {
            type = types.nullOr types.bool;
            default = null;
            description = "Whether to honor gitignore for this component.";
          };
        };
      });
    in
      types.submodule ({ ... }: {
        freeformType = manifestFreeformType;
        options = {
          config = mkOption {
            type = manifestFreeformType;
            default = {};
            description = "Manifest config mapping.";
          };

          components = mkOption {
            type = types.listOf componentType;
            description = "Manifest components.";
          };
        };
      });
  managedContextType = types.submodule ({ ... }: {
    options = {
      targetDir = mkOption {
        type = types.str;
        description = "Directory where the context should be hydrated.";
      };

      replace = mkOption {
        type = types.enum [ "guarded" "always" "never" ];
        default = "guarded";
        description = "Replacement policy for an existing context directory.";
      };

      manifest = {
        source = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Path to a manifest source. Relative paths are resolved from targetDir.";
        };

        text = mkOption {
          type = types.nullOr types.str;
          default = null;
          description = "Inline manifest source text.";
        };

        data = mkOption {
          type = types.nullOr manifestDataType;
          default = null;
          description = "Nix-declared manifest data.";
        };
      };
    };
  });
  manifestFor = context:
    if context.manifest.source != null then
      { source = context.manifest.source; }
    else if context.manifest.text != null then
      { text = context.manifest.text; }
    else
      { data = cleanNulls context.manifest.data; };
  registry = {
    version = 1;
    contexts = lib.mapAttrs (_name: context: {
      targetDir = context.targetDir;
      replace = context.replace;
      manifest = manifestFor context;
    }) cfg.managedContexts;
  };
  registryFile = pkgs.writeText "contextualize-managed-contexts.json" (
    builtins.toJSON registry
  );
  hasManagedContexts = cfg.managedContexts != {};
  effectiveManagedActivationEnable =
    if cfg.managedActivation.enable == null then hasManagedContexts else cfg.managedActivation.enable;
  managedActivationMode = cfg.managedActivation.mode;
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
  managedActivationArgs =
    [ "hydrate" "--managed" "--managed-registry" "${registryFile}" ]
    ++ lib.optional cfg.managedActivation.strict "--strict";
  managedActivationCommand =
    lib.escapeShellArgs ([ "${wrapper}/bin/contextualize" ] ++ managedActivationArgs);
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

    runtimePackages = lib.mkOption {
      type = lib.types.listOf lib.types.package;
      default = [ pkgs.git ];
      description = "Packages placed on PATH when running contextualize through the managed wrapper.";
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

    managedContexts = mkOption {
      type = types.attrsOf managedContextType;
      default = {};
      description = "Named context manifests to hydrate from the managed registry.";
    };

    managedActivation = {
      enable = mkOption {
        type = types.nullOr types.bool;
        default = null;
        description = "Hydrate managed contexts during Home Manager activation.";
      };

      mode = mkOption {
        type = types.enum [ "service" "inline" ];
        default = if pkgs.stdenv.isLinux then "service" else "inline";
        description = "Run managed hydration through a non-blocking user service or inline during activation.";
      };

      after = mkOption {
        type = types.listOf types.str;
        default = [ "writeBoundary" ];
        description = "Home Manager activation steps that must run before managed hydration.";
      };

      serviceTimeout = mkOption {
        type = types.str;
        default = "30min";
        description = "Timeout for the managed hydration user service.";
      };

      strict = mkOption {
        type = types.bool;
        default = false;
        description = "Fail inline activation or the managed hydration service when a managed context fails.";
      };
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = lib.mapAttrsToList (name: context:
      let
        setSources = lib.filter (value: value != null) [
          context.manifest.source
          context.manifest.text
          context.manifest.data
        ];
      in
      {
        assertion = lib.length setSources == 1;
        message = "programs.contextualize.managedContexts.${name}.manifest must set exactly one of source, text, or data";
      }
    ) cfg.managedContexts;

    home.packages = [ wrapper ];

    xdg.configFile."contextualize/managed-contexts.json" = lib.mkIf hasManagedContexts {
      source = registryFile;
    };

    systemd.user.services.contextualize-managed-contexts = lib.mkIf (
      pkgs.stdenv.isLinux
      && effectiveManagedActivationEnable
      && hasManagedContexts
      && managedActivationMode == "service"
    ) {
      Unit = {
        Description = "Hydrate managed contextualize manifests";
      };

      Service = {
        Type = "oneshot";
        ExecStart = managedActivationCommand;
        TimeoutStartSec = cfg.managedActivation.serviceTimeout;
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

    home.activation.contextualizeManagedContexts = lib.mkIf (effectiveManagedActivationEnable && hasManagedContexts) (
      lib.hm.dag.entryAfter (
        cfg.managedActivation.after
        ++ lib.optional (managedActivationMode == "service") "reloadSystemd"
      ) ''
        if [ -n "''${DRY_RUN_CMD:-}" ]; then
          echo "contextualize: would hydrate managed contexts"
        elif [ ${lib.escapeShellArg managedActivationMode} = service ]; then
          ${pkgs.systemd}/bin/systemctl --user start --no-block contextualize-managed-contexts.service >/dev/null 2>&1 || true
        else
          ${managedActivationCommand}
        fi
      ''
    );
  };
}
