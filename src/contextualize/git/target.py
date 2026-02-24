import os
import re
from dataclasses import dataclass
from urllib.parse import quote, unquote, urlparse, urlunparse

CACHE_ROOT = os.path.expanduser("~/.local/share/contextualize/cache/git")


@dataclass
class GitTarget:
    repo_url: str
    cache_dir: str
    path: str | None
    rev: str | None


def _get_host_and_repo(url: str) -> tuple[str, str]:
    if url.startswith("git@"):
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
    repo_url, rev, path = target, None, None

    if ":" in target:
        if target.startswith("git@"):
            first_colon = target.find(":")
            if first_colon != -1:
                depth = 0
                seen_slash = False
                colon_pos = -1
                for i in range(first_colon + 1, len(target)):
                    ch = target[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth = max(0, depth - 1)
                    elif ch == "/":
                        seen_slash = True
                    elif ch == ":" and depth == 0 and seen_slash:
                        colon_pos = i
                        break
                if colon_pos != -1 and colon_pos + 1 < len(target):
                    path = target[colon_pos + 1 :]
                    repo_url = target[:colon_pos]
        else:
            protocol_end = target.find("://")
            if protocol_end != -1:
                slash_after_host = target.find("/", protocol_end + 3)
                if slash_after_host != -1:
                    depth = 0
                    colon_pos = -1
                    for i in range(slash_after_host, len(target)):
                        ch = target[i]
                        if ch == "{":
                            depth += 1
                            continue
                        if ch == "}":
                            depth = max(0, depth - 1)
                            continue
                        if depth == 0 and target.startswith("://", i):
                            break
                        if ch == ":" and depth == 0:
                            colon_pos = i
                            break
                    if colon_pos != -1 and colon_pos + 1 < len(target):
                        path = target[colon_pos + 1 :]
                        repo_url = target[:colon_pos]

    if "#" in repo_url:
        hash_pos = repo_url.rfind("#")
        potential_rev = repo_url[hash_pos + 1 :]
        if potential_rev and "/" not in potential_rev:
            rev, repo_url = potential_rev, repo_url[:hash_pos]

    if "@" in repo_url:
        if repo_url.startswith("git@"):
            git_at_end = repo_url.find(":", 4)
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
            at_pos = repo_url.rfind("@")
            potential_rev = repo_url[at_pos + 1 :]
            if potential_rev and "/" not in potential_rev:
                if not rev:
                    rev = potential_rev
                repo_url = repo_url[:at_pos]

    parsed_repo = urlparse(repo_url)
    if parsed_repo.scheme in {"http", "https"}:
        repo_url = urlunparse(
            (
                parsed_repo.scheme,
                parsed_repo.netloc,
                parsed_repo.path,
                "",
                "",
                "",
            )
        )
        repo_url, rev, path = _normalize_github_web_target(repo_url, rev, path)

    return repo_url, rev, path


def _normalize_github_web_target(
    repo_url: str, rev: str | None, path: str | None
) -> tuple[str, str | None, str | None]:
    parsed = urlparse(repo_url)
    host = (parsed.hostname or "").lower()
    if host not in {"github.com", "www.github.com"}:
        return repo_url, rev, path

    segments = [part for part in parsed.path.split("/") if part]
    if len(segments) < 4:
        return repo_url, rev, path
    if segments[2] not in {"blob", "tree"}:
        return repo_url, rev, path

    owner, repo = segments[0], segments[1]
    normalized_repo_url = urlunparse(
        (parsed.scheme, parsed.netloc, f"/{owner}/{repo}", "", "", "")
    )
    if rev and re.fullmatch(r"L\d+(?:-L\d+)?", rev):
        normalized_rev = segments[3]
    else:
        normalized_rev = rev or segments[3]
    normalized_path = path
    if normalized_path is None and len(segments) > 4:
        normalized_path = "/".join(unquote(segment) for segment in segments[4:])
    elif normalized_path is not None:
        normalized_path = unquote(normalized_path)

    return normalized_repo_url, normalized_rev, normalized_path


def github_blob_to_raw_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in {"github.com", "www.github.com"}:
        return None

    segments = [part for part in parsed.path.split("/") if part]
    if len(segments) < 5 or segments[2] != "blob":
        return None

    owner = unquote(segments[0])
    repo = unquote(segments[1])
    rev = quote(unquote(segments[3]), safe="")
    raw_path = "/".join(unquote(segment) for segment in segments[4:])
    encoded_path = "/".join(quote(part, safe="") for part in raw_path.split("/"))
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{rev}/{encoded_path}"


def _is_supported_git_http_url(
    repo_url: str, *, has_explicit_ref: bool, has_explicit_path: bool
) -> bool:
    parsed = urlparse(repo_url)
    if parsed.scheme not in {"http", "https"}:
        return False

    host = (parsed.hostname or "").lower()
    segments = [part for part in parsed.path.split("/") if part]
    if not segments:
        return False
    if segments[-1].endswith(".git"):
        return True

    known_hosts = {
        "github.com",
        "gitlab.com",
        "bitbucket.org",
        "git.sr.ht",
        "www.github.com",
        "www.gitlab.com",
        "www.bitbucket.org",
    }
    if host in known_hosts:
        return len(segments) == 2

    if has_explicit_ref or has_explicit_path:
        return len(segments) >= 2

    return False


def parse_git_target(target: str) -> GitTarget | None:
    from ..references.helpers import parse_target_spec

    target = parse_target_spec(target).get("target", target)
    for prefix in ("github:", "gh:"):
        if target.startswith(prefix):
            rest = target[len(prefix) :]
            repo_part, _, path = rest.partition(":")
            repo, _, rev = repo_part.partition("@")
            repo = repo[:-4] if repo.endswith(".git") else repo
            repo_url = f"git@github.com:{repo}.git"
            cache_dir = os.path.join(CACHE_ROOT, "github", *repo.split("/"))
            return GitTarget(repo_url, cache_dir, path or None, rev or None)

    repo_url, rev, path = _extract_path_and_rev(target)

    if not (repo_url.startswith("http") or repo_url.startswith("git@")):
        return None
    if repo_url.startswith("http") and not _is_supported_git_http_url(
        repo_url,
        has_explicit_ref=rev is not None,
        has_explicit_path=path is not None,
    ):
        return None

    host, repo = _get_host_and_repo(repo_url)
    root = "github" if host.endswith("github.com") else os.path.join("ext", host)
    cache_dir = os.path.join(CACHE_ROOT, root, *repo.split("/"))
    return GitTarget(repo_url, cache_dir, path, rev)
