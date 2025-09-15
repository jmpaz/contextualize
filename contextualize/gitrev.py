import os
import subprocess
from dataclasses import dataclass
from typing import Iterable, List, Optional

from .reference import process_text


def _run_git(repo_root: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", repo_root, *args], check=True, capture_output=True, text=True
    )


def get_repo_root(start_path: str) -> Optional[str]:
    path = start_path if not os.path.isfile(start_path) else os.path.dirname(start_path)
    try:
        return _run_git(path, ["rev-parse", "--show-toplevel"]).stdout.strip()
    except subprocess.CalledProcessError:
        return None


def _to_rel(repo_root: str, p: str) -> str:
    if not os.path.isabs(p):
        p = os.path.abspath(p)
    repo_root = os.path.abspath(repo_root)
    if os.path.commonpath([repo_root, p]) != repo_root:
        raise ValueError(f"Path is outside the repository: {p}")
    return os.path.relpath(p, repo_root)


def list_files_at_rev(repo_root: str, rev: str, path_specs: Iterable[str]) -> List[str]:
    files: list[str] = []
    seen: set[str] = set()
    for spec in path_specs:
        spec_rel = (
            _to_rel(repo_root, spec)
            if os.path.isabs(spec) or str(spec).startswith("..")
            else spec
        )
        try:
            out = _run_git(
                repo_root, ["ls-tree", "-r", "--name-only", rev, "--", spec_rel]
            ).stdout
        except subprocess.CalledProcessError:
            continue
        for line in out.splitlines():
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                files.append(line)
    return files


def read_file_at_rev(repo_root: str, rev: str, rel_path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "show", f"{rev}:{rel_path}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return None


@dataclass
class GitRevFileReference:
    repo_root: str
    rev: str
    rel_path: str
    format: str = "md"
    label: str = "relative"

    def get_label(self) -> str:
        if self.label == "relative":
            return self.rel_path
        if self.label == "name":
            return os.path.basename(self.rel_path)
        if self.label == "ext":
            return os.path.splitext(self.rel_path)[1]
        return self.label

    @property
    def output(self) -> str:
        text = read_file_at_rev(self.repo_root, self.rev, self.rel_path)
        if text is None:
            return ""
        return process_text(
            text, format=self.format, label=self.get_label(), rev=self.rev
        )
