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


def _extract_path_and_rev(target: str) -> tuple[str, str | None, str | None]:
    """Extract path and revision from target, returning (url, rev, path)."""
    repo_url, rev, path = target, None, None

    # path specifier
    if ":" in target:
        if target.startswith("git@"):
            # git@host:owner/repo
            first_colon = target.find(":")
            if target.count(":") > 1:
                remainder = target[first_colon + 1 :]
                last_colon = remainder.rfind(":")
                if last_colon != -1 and remainder[last_colon + 1 :]:
                    path = remainder[last_colon + 1 :]
                    repo_url = target[: first_colon + 1 + last_colon]
        else:
            # http/https URLs - colon after protocol
            colon_pos = target.rfind(":")
            protocol_end = target.find("://")
            if protocol_end != -1 and colon_pos > protocol_end + 2:
                potential_path = target[colon_pos + 1 :]
                if potential_path and (
                    not potential_path.isdigit() or "/" in potential_path
                ):
                    path, repo_url = potential_path, target[:colon_pos]

    # revision specifier
    # handle # first (takes precedence for specific commit hashes)
    if "#" in repo_url:
        hash_pos = repo_url.rfind("#")
        potential_rev = repo_url[hash_pos + 1 :]
        if potential_rev and "/" not in potential_rev:
            rev, repo_url = potential_rev, repo_url[:hash_pos]

    # handle @ for branch/revision
    if "@" in repo_url:
        if repo_url.startswith("git@"):
            # for git@ URLs, only look for @ after the first one
            git_at_end = repo_url.find(":", 4)  # find colon after git@host
            if git_at_end != -1:
                remaining = repo_url[git_at_end + 1 :]
                if "@" in remaining:
                    at_pos = remaining.rfind("@")
                    potential_rev = remaining[at_pos + 1 :]
                    if potential_rev and "/" not in potential_rev:
                        if not rev:
                            rev = potential_rev
                        repo_url = repo_url[: git_at_end + 1 + at_pos]
        else:
            # regular URLs
            at_pos = repo_url.rfind("@")
            potential_rev = repo_url[at_pos + 1 :]
            if potential_rev and "/" not in potential_rev:
                if not rev:
                    rev = potential_rev
                repo_url = repo_url[:at_pos]

    # clean up .git suffix
    if repo_url.startswith("http") and repo_url.endswith(".git"):
        repo_url = repo_url[:-4]

    return repo_url, rev, path


def parse_git_target(target: str) -> GitTarget | None:
    if target.startswith("gh:"):
        rest = target[3:]
        repo_part, _, path = rest.partition(":")
        repo, _, rev = repo_part.partition("@")
        repo = repo[:-4] if repo.endswith(".git") else repo
        repo_url = f"git@github.com:{repo}.git"
        cache_dir = os.path.join(CACHE_ROOT, "github", *repo.split("/"))
        return GitTarget(repo_url, cache_dir, path or None, rev or None)

    repo_url, rev, path = _extract_path_and_rev(target)

    if not (repo_url.startswith("http") or repo_url.startswith("git@")):
        return None

    host, repo = _get_host_and_repo(repo_url)
    root = "github" if host.endswith("github.com") else os.path.join("ext", host)
    cache_dir = os.path.join(CACHE_ROOT, root, *repo.split("/"))
    return GitTarget(repo_url, cache_dir, path, rev)


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


def _split_brace_options(s: str) -> list[str]:
    opts = []
    buf = ""
    depth = 0
    for ch in s:
        if ch == "," and depth == 0:
            opts.append(buf)
            buf = ""
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        buf += ch
    opts.append(buf)
    return opts


def brace_expand(pattern: str) -> list[str]:
    m = re.search(r"\{", pattern)
    if not m:
        return [pattern]
    start = m.start()
    depth = 0
    end = start
    for i in range(start, len(pattern)):
        if pattern[i] == "{":
            depth += 1
        elif pattern[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    inside = pattern[start + 1 : end]
    rest = pattern[end + 1 :]
    prefix = pattern[:start]
    out = []
    for opt in _split_brace_options(inside):
        for expanded in brace_expand(opt + rest):
            out.append(prefix + expanded)
    return out


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
