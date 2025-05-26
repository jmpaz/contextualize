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

    m = re.match(r"(?P<url>(?:https?://|git@)[^:@]+(?::[0-9]+)?/[^:@]+?)(?:\.git)?(?:@(?P<rev>[^:]+))?(?::(?P<path>.*))?$", target)
    if not m:
        return None
    repo_url = m.group("url")
    if not repo_url.endswith(".git"):
        repo_url += ".git"
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
        subprocess.run(["git", "clone", "--depth", "1", g.repo_url, g.cache_dir], check=False)
    elif pull:
        subprocess.run(["git", "-C", g.cache_dir, "pull"], check=False)
    if g.rev:
        subprocess.run(["git", "-C", g.cache_dir, "fetch", "--depth", "1", "origin", g.rev], check=False)
        subprocess.run(["git", "-C", g.cache_dir, "checkout", g.rev], check=False)
    return g.cache_dir
