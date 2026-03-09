from contextualize.git.target import parse_git_target


def test_parse_git_target_rejects_http_fragment_on_non_git_host() -> None:
    assert parse_git_target("https://en.wikipedia.org/wiki/Wikipedia#History") is None


def test_parse_git_target_rejects_http_path_suffix_on_non_git_host() -> None:
    assert parse_git_target("https://en.wikipedia.org/wiki/Wikipedia:About") is None


def test_parse_git_target_keeps_dot_git_urls_on_unknown_hosts() -> None:
    target = parse_git_target(
        "https://git.example.com/org/repo.git@main:path/in/repo.txt"
    )

    assert target is not None
    assert target.repo_url == "https://git.example.com/org/repo.git"
    assert target.rev == "main"
    assert target.path == "path/in/repo.txt"
