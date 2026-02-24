import os
import re
import shutil
import subprocess

from ..utils import _split_brace_options, brace_expand
from . import target as _target

CACHE_ROOT = _target.CACHE_ROOT
GitTarget = _target.GitTarget
parse_git_target = _target.parse_git_target


def ensure_repo(g: GitTarget, pull: bool = False, reclone: bool = False) -> str:
    if reclone and os.path.isdir(g.cache_dir):
        shutil.rmtree(g.cache_dir)

    if not os.path.isdir(g.cache_dir):
        os.makedirs(os.path.dirname(g.cache_dir), exist_ok=True)

        # only use --depth 1 if we're not targeting a specific revision
        def do_clone(url: str) -> None:
            clone_args = ["git", "clone"]
            if not g.rev:
                clone_args.extend(["--depth", "1"])
            clone_args.extend([url, g.cache_dir])
            subprocess.run(clone_args, check=True, capture_output=True)

        try:
            do_clone(g.repo_url)
        except subprocess.CalledProcessError as err:
            alt_url = None
            if g.repo_url.endswith(".git"):
                alt_url = g.repo_url[:-4]
            else:
                alt_url = g.repo_url + ".git"

            try:
                do_clone(alt_url)
                g.repo_url = alt_url
            except subprocess.CalledProcessError:
                raise err
    elif pull:
        if g.rev:
            subprocess.run(
                ["git", "-C", g.cache_dir, "fetch", "origin"],
                check=True,
                capture_output=True,
            )
        else:
            subprocess.run(
                ["git", "-C", g.cache_dir, "pull"], check=True, capture_output=True
            )

    if g.rev:
        rev = g.rev
        rev_is_hash = re.fullmatch(r"[0-9a-f]{6,40}", rev) is not None
        # if this looks like a short hash, resolve it to full hash
        if re.fullmatch(r"[0-9a-f]{6,39}", rev):
            try:
                out = subprocess.run(
                    ["git", "ls-remote", g.repo_url, rev + "*"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in out.stdout.splitlines():
                    sha = line.split("\t", 1)[0]
                    if sha.startswith(rev):
                        rev = sha
                        break
            except subprocess.CalledProcessError:
                # if ls-remote fails, try the original rev as-is
                pass

        checkout_targets = [rev]
        if pull and not rev_is_hash and not rev.startswith("refs/"):
            checkout_targets.insert(0, f"origin/{rev}")

        checkout_error: subprocess.CalledProcessError | None = None
        for candidate in checkout_targets:
            try:
                subprocess.run(
                    ["git", "-C", g.cache_dir, "checkout", candidate],
                    check=True,
                    capture_output=True,
                )
                checkout_error = None
                break
            except subprocess.CalledProcessError as err:
                checkout_error = err
                continue

        if checkout_error is not None:
            try:
                subprocess.run(
                    ["git", "-C", g.cache_dir, "fetch", "origin", f"{rev}:{rev}"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(
                        ["git", "-C", g.cache_dir, "fetch", "--unshallow"],
                        check=True,
                        capture_output=True,
                    )
                except subprocess.CalledProcessError:
                    subprocess.run(
                        [
                            "git",
                            "-C",
                            g.cache_dir,
                            "fetch",
                            "origin",
                            "+refs/heads/*:refs/remotes/origin/*",
                        ],
                        check=True,
                        capture_output=True,
                    )

            subprocess.run(
                ["git", "-C", g.cache_dir, "checkout", rev],
                check=True,
                capture_output=True,
            )
    else:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    g.cache_dir,
                    "symbolic-ref",
                    "--quiet",
                    "refs/remotes/origin/HEAD",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            default_branch = result.stdout.strip().split("/")[-1]
            subprocess.run(
                ["git", "-C", g.cache_dir, "checkout", default_branch],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            for branch in ["main", "master"]:
                try:
                    subprocess.run(
                        ["git", "-C", g.cache_dir, "checkout", branch],
                        check=True,
                        capture_output=True,
                    )
                    break
                except subprocess.CalledProcessError:
                    continue

    return g.cache_dir


def expand_git_paths(repo_dir: str, spec: str) -> list[str]:
    from glob import glob

    paths: list[str] = []
    for part in _split_brace_options(spec):
        for expanded_path in brace_expand(part):
            full = os.path.join(repo_dir, expanded_path)
            matches = glob(full, recursive=True)
            if matches:
                paths.extend(matches)
            else:
                paths.append(full)
    return paths
