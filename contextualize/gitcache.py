import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

CACHE_ROOT = os.path.expanduser("~/.local/share/contextualize/cache/git")


@dataclass
class GitTarget:
    repo_url: str
    cache_dir: str
    path: str | None
    rev: str | None


def _get_host_and_repo(url: str) -> tuple[str, str]:
    if url.startswith("git@"):  # git@github.com:owner/repo.git
        host_path = url[4:]
        host, path = host_path.split(":", 1)
    else:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        path = parsed.path.lstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    return host, path


def parse_git_target(target: str) -> GitTarget | None:
    if target.startswith("gh:"):
        rest = target[3:]
        repo_part, _, path = rest.partition(":")
        repo, _, rev = repo_part.partition("@")
        repo = repo[:-4] if repo.endswith(".git") else repo
        repo_url = f"git@github.com:{repo}.git"
        cache_dir = os.path.join(CACHE_ROOT, "github", *repo.split("/"))
        return GitTarget(repo_url, cache_dir, path or None, rev or None)

    m = re.match(
        r"(?P<url>(?:https?://|git@)[^:@]+(?::[0-9]+)?/[^:@]+?)(?:\.git)?(?:@(?P<rev>[^:]+))?(?::(?P<path>.*))?$",
        target,
    )
    if not m:
        return None
    repo_url = m.group("url")
    rev = m.group("rev")
    path = m.group("path")
    host, repo = _get_host_and_repo(repo_url)
    root = "github" if host.endswith("github.com") else os.path.join("ext", host)
    cache_dir = os.path.join(CACHE_ROOT, root, *repo.split("/"))
    return GitTarget(repo_url, cache_dir, path or None, rev or None)


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
        subprocess.run(
            ["git", "-C", g.cache_dir, "pull"], check=True, capture_output=True
        )

    if g.rev:
        rev = g.rev
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

        # try to checkout first - it might already be available
        try:
            subprocess.run(
                ["git", "-C", g.cache_dir, "checkout", rev],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # if checkout fails, we might have a shallow repo that needs more history
            try:
                # first try to unshallow the repo
                subprocess.run(
                    ["git", "-C", g.cache_dir, "fetch", "--unshallow"],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                # if unshallow fails (maybe it's already unshallow), just fetch all
                subprocess.run(
                    ["git", "-C", g.cache_dir, "fetch", "origin"],
                    check=True,
                    capture_output=True,
                )

            # now try to checkout again
            subprocess.run(
                ["git", "-C", g.cache_dir, "checkout", rev],
                check=True,
                capture_output=True,
            )
    else:
        # no specific revision, checkout default branch
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
            # if symbolic-ref fails, fall back to main/master
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
