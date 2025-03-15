import os
import re
from urllib.parse import urljoin, urlparse

import click
import requests


class Crawler:
    """
    A recursive crawler for .txt and .md URLs.

    - If a URL ends with '.txt', the content is scanned for markdown links and those linked files are crawled recursively.
    - If a URL ends with '.md', it is downloaded without further link scanning.
    - All fetched content is cached in a local cache (default: ~/.local/share/contextualize/cache)
      unless the --refresh-cache flag is set.
    - By default, only same-domain links are followed (compare to the top-level domain(s)).
    - A --follow-external flag allows including external domains.
    - A --max-depth parameter limits how many layers of links we crawl.
    """

    def __init__(
        self,
        cache_dir=None,
        cache_refresh=False,
        follow_external=False,
        max_depth=2,
    ):
        if cache_dir is None:
            home = os.path.expanduser("~")
            self.cache_dir = os.path.join(
                home, ".local", "share", "contextualize", "cache"
            )
        else:
            self.cache_dir = cache_dir

        self.cache_refresh = cache_refresh
        self.follow_external = follow_external
        self.max_depth = max_depth

        self.visited = set()  # URLs that have already been crawled
        self.url_to_content = {}  # Map: URL -> downloaded content
        self.top_level_domains = set()

        os.makedirs(self.cache_dir, exist_ok=True)

    def crawl(self, url, depth=1):
        """
        Crawl the given URL if valid. For .txt files, recursively scan for markdown links.

        :param url: The URL to fetch
        :param depth: Current recursion depth
        """
        if depth > self.max_depth:
            return

        if not self._is_valid_url(url):
            return

        parsed = urlparse(url)
        # If this is a "top-level" crawl call, store its domain
        # so we can limit subsequent crawls to that domain, if follow_external=False
        if depth == 1:
            self.top_level_domains.add(parsed.netloc.lower())

        # Check domain restrictions
        if not self.follow_external:
            # Only follow if domain is in top_level_domains
            if parsed.netloc.lower() not in self.top_level_domains:
                return

        # skip if visited
        if url in self.visited:
            return
        self.visited.add(url)

        # fetch
        content = self._fetch_and_cache(url)
        if not content:
            return

        self.url_to_content[url] = content

        # If the file is a .txt, extract further markdown links and crawl them
        if url.lower().endswith(".txt"):
            links = self._extract_links(content)
            for link in links:
                full_link = urljoin(url, link)
                self.crawl(full_link, depth=depth + 1)

    def to_references(self, wrap_mode=None):
        """
        Combine all crawled files into a single output.

        If wrap_mode == "xml":
          <context>
            <ctx path="...">
              ...
            </ctx>
            ...
          </context>
        If wrap_mode == "md":
          ```context
          ```...some-file.txt
          content
          ```
          ```another-file.md
          content
          ```
          ```  # end
        Else:
          === label ===
          content
          === label 2 ===
          content
        """
        if wrap_mode == "xml":
            ref_lines = ["<context>"]
            for url, content in self.url_to_content.items():
                label = self._label_from_url(url)
                ref_block = f'<ctx path="{label}">\n{content}\n</ctx>'
                ref_lines.append(ref_block)
            ref_lines.append("</context>")
            return "\n".join(ref_lines)

        elif wrap_mode == "md":
            ref_blocks = []
            for url, content in self.url_to_content.items():
                label = self._label_from_url(url)
                block = self._format_markdown_reference(label, content)
                ref_blocks.append(block)
            combined = "\n\n".join(ref_blocks)
            outer_fence = self._generate_fence(combined, language="context")
            return f"{outer_fence}\n{combined}\n{outer_fence}"

        else:
            # plain text
            ref_lines = []
            for url, content in self.url_to_content.items():
                label = self._label_from_url(url)
                block = f"=== {label} ===\n{content}"
                ref_lines.append(block)
            return "\n\n".join(ref_lines)

    def _label_from_url(self, url):
        """
        Construct a label from the URL path. If a path exists, remove the leading slash.
        Otherwise fall back to the netloc or the URL itself.
        """
        parsed = urlparse(url)
        if parsed.path.strip("/"):
            return parsed.path.lstrip("/")
        return parsed.netloc or url

    def _is_valid_url(self, url):
        """
        Check that the URL ends with .txt or .md (case-insensitive).
        """
        lower = url.lower()
        return lower.endswith(".txt") or lower.endswith(".md")

    def _fetch_and_cache(self, url):
        """
        Download content from the URL and cache it locally.
        If the cache exists (and --refresh-cache was not used), read from disk.
        """
        parsed = urlparse(url)
        # create a safe filename
        safe_path = re.sub(r"[^\w\.-]+", "_", parsed.path or "root")
        if not os.path.splitext(safe_path)[1]:
            # if there's no extension, force .txt
            safe_path += ".txt"
        cache_file = os.path.join(self.cache_dir, f"{parsed.netloc}_{safe_path}")

        if not os.path.exists(cache_file) or self.cache_refresh:
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                with open(cache_file, "w", encoding="utf-8") as f:
                    f.write(resp.text)
            except Exception as e:
                click.echo(f"Error fetching {url}: {str(e)}", err=True)
                return ""

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            click.echo(f"Error reading cache {cache_file}: {str(e)}", err=True)
            return ""

    def _extract_links(self, content):
        """
        Extract markdown links (with .txt or .md) from the given content.
        e.g. [API List](https://fastht.ml/docs/apilist.txt): optional text
        """
        pattern = r"\[.*?\]\((https?:\/\/[^\s)]+(\.txt|\.md))\)"
        matches = re.findall(pattern, content, flags=re.IGNORECASE)
        return [m[0] for m in matches]

    def _generate_fence(self, text, language=""):
        """
        Generate a markdown fence that is longer than any sequence of backticks found in 'text'.
        Minimum length is 3. Optionally append a language tag (e.g. ```python).
        """
        max_backticks = 0
        for match in re.findall(r"`+", text):
            max_backticks = max(max_backticks, len(match))
        fence_len = max(max_backticks + 1, 3)
        fence = "`" * fence_len
        if language:
            fence += language
        return fence

    def _format_markdown_reference(self, label, content):
        """
        Format a single reference block as a fenced code block. The fence length
        is chosen to exceed any backticks in 'content'. 'label' is used
        as the 'language specifier', e.g. ```filename
        """
        fence = self._generate_fence(content, language=label)
        return f"{fence}\n{content}\n{fence}"
