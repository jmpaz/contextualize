"""Content injection system for contextualize.

Supports parsing and replacing {cx::...} patterns with content from:
- HTTP URLs (raw content fetching)
- Git repositories (using existing gitcache.py)
- Local files (using existing reference.py)
"""

import os
import re
import subprocess
from typing import Dict, Optional, Union
from urllib.parse import urlparse

from .gitcache import ensure_repo, expand_git_paths, parse_git_target
from .reference import create_file_references


class InjectionParser:
    """Parses {cx::...} injection patterns from text content."""

    # Regex to match {cx::...} patterns with optional parameters
    INJECTION_PATTERN = re.compile(r"\{cx::([^}]+)\}")

    def __init__(self):
        pass

    def parse_injection(self, match: re.Match) -> Dict[str, Union[str, None]]:
        """Parse an injection match into components."""
        content = match.group(1)

        # Split on :: to separate parameters from target
        parts = content.split("::")

        # Parse parameters from the beginning
        params = {}
        target_start_idx = 0

        for i, part in enumerate(parts):
            # Check if this part looks like a parameter (key="value" or key=value)
            param_match = re.match(
                r'^(filename|params|root)=(?:"([^"]*)"|([^"]*))$', part
            )
            if param_match:
                key = param_match.group(1)
                # Use quoted value if present, otherwise unquoted value
                value = (
                    param_match.group(2)
                    if param_match.group(2) is not None
                    else param_match.group(3)
                )
                params[key] = value
                target_start_idx = i + 1
            else:
                # Once we hit a non-parameter part, everything from here is the target
                break

        # Everything from target_start_idx onward is the target
        target_parts = parts[target_start_idx:]
        target = "::".join(target_parts) if target_parts else content

        return {
            "target": target,
            "filename": params.get("filename"),
            "params": params.get("params"),
            "root": params.get("root"),
            "full_match": match.group(0),
        }

    def is_http_url(self, target: str) -> bool:
        """Check if target is an HTTP/HTTPS URL."""
        parsed = urlparse(target)
        return parsed.scheme in ("http", "https")

    def is_git_target(self, target: str) -> bool:
        """Check if target is a git repository reference."""
        return any(
            [
                target.startswith("git@"),
                target.startswith("https://")
                and any(
                    host in target
                    for host in ["github.com", "gitlab.com", "bitbucket.org"]
                ),
                target.startswith("gh:"),
                "@" in target and ":" in target,  # git@host:owner/repo pattern
            ]
        )

    def fetch_http_content(self, url: str, filename: Optional[str] = None) -> str:
        """Fetch content from HTTP URL."""
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests library required for HTTP fetching. Install with: pip install requests"
            )

        try:
            response = requests.get(
                url, timeout=30, headers={"User-Agent": "contextualize/0.1.0"}
            )
            response.raise_for_status()

            # Use filename for labeling if provided, otherwise use URL
            label = filename or url

            # Handle different content types
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" in content_type:
                # Pretty-print JSON
                import json

                try:
                    parsed = response.json()
                    content = json.dumps(parsed, indent=2)
                except json.JSONDecodeError:
                    content = response.text
            else:
                content = response.text

            # Format like normal cat output using proper markdown code fences
            from .reference import process_text

            return process_text(content, format="md", label=label)

        except requests.exceptions.Timeout:
            raise Exception(
                f"Timeout fetching {url}: Request timed out after 30 seconds"
            )
        except requests.exceptions.ConnectionError:
            raise Exception(f"Connection failed for {url}: Could not connect to server")
        except requests.exceptions.HTTPError as e:
            raise Exception(
                f"HTTP {e.response.status_code} for {url}: {e.response.reason}"
            )
        except Exception as e:
            raise Exception(f"Error fetching {url}: {str(e)}")

    def fetch_git_content(self, target: str, params: Optional[str] = None) -> str:
        """Fetch content from git repository using existing gitcache."""
        # Parse git target to extract repo and path info
        git_target = parse_git_target(target)
        if not git_target:
            raise Exception(f"Invalid git target format: {target}")

        # Parse params to extract git options
        git_pull = False
        git_reclone = False
        output_format = "md"
        label_style = "relative"

        if params:
            # Parse command-line style parameters
            param_parts = params.split()
            i = 0
            while i < len(param_parts):
                param = param_parts[i]
                if param == "--git-pull":
                    git_pull = True
                elif param == "--git-reclone":
                    git_reclone = True
                elif param == "--format" and i + 1 < len(param_parts):
                    output_format = param_parts[i + 1]
                    i += 1
                elif param == "--label" and i + 1 < len(param_parts):
                    label_style = param_parts[i + 1]
                    i += 1
                i += 1

        try:
            # Get the repository content with appropriate options
            repo_path = ensure_repo(git_target, pull=git_pull, reclone=git_reclone)

            if not repo_path:
                raise Exception(f"Failed to fetch git repository: {target}")

            # Handle path specification
            if git_target.path:
                # Use existing expand logic for brace expansion
                target_paths = expand_git_paths(repo_path, git_target.path)
                if len(target_paths) == 1 and os.path.isfile(target_paths[0]):
                    # Single file
                    with open(target_paths[0], "r", encoding="utf-8") as f:
                        content = f.read()
                    from .reference import process_text

                    return process_text(
                        content, format=output_format, label=git_target.path
                    )
                else:
                    # Multiple files or directory - use existing file gathering logic
                    refs = create_file_references(
                        target_paths, format=output_format, label=label_style
                    )
                    return refs["concatenated"]
            else:
                # No specific path - return repository root contents
                refs = create_file_references(
                    [repo_path], format=output_format, label=label_style
                )
                return refs["concatenated"]

        except subprocess.CalledProcessError as e:
            error_output = e.stderr.decode("utf-8") if e.stderr else "No error output"
            raise Exception(
                f"Git operation failed for {target}: {e.cmd} (exit {e.returncode}): {error_output}"
            )
        except Exception as e:
            raise Exception(f"Error fetching git content {target}: {str(e)}")

    def fetch_local_content(
        self, target: str, root: Optional[str] = None, params: Optional[str] = None
    ) -> str:
        """Fetch content from local file system."""
        # Parse params to extract options
        output_format = "md"
        label_style = "relative"

        if params:
            # Parse command-line style parameters
            param_parts = params.split()
            i = 0
            while i < len(param_parts):
                param = param_parts[i]
                if param == "--format" and i + 1 < len(param_parts):
                    output_format = param_parts[i + 1]
                    i += 1
                elif param == "--label" and i + 1 < len(param_parts):
                    label_style = param_parts[i + 1]
                    i += 1
                i += 1

        # Handle root path specification
        if root:
            expanded_root = os.path.expanduser(root)
            if not os.path.isdir(expanded_root):
                raise Exception(f"Root directory not found: {expanded_root}")
            base_path = expanded_root
            full_target = os.path.join(expanded_root, target)
        else:
            base_path = os.getcwd()
            full_target = os.path.expanduser(target)

        try:
            # Handle brace expansion for local files
            from glob import glob

            if "{" in target and "}" in target:
                # Use glob-style brace expansion
                if root:
                    pattern = os.path.join(base_path, target)
                else:
                    pattern = target

                expanded_paths = []
                # Simple brace expansion - split on commas within braces
                import re

                brace_match = re.search(r"\{([^}]+)\}", target)
                if brace_match:
                    options = brace_match.group(1).split(",")
                    base_pattern = (
                        target[: brace_match.start()]
                        + "{}"
                        + target[brace_match.end() :]
                    )
                    for option in options:
                        expanded_target = base_pattern.format(option.strip())
                        if root:
                            path = os.path.join(base_path, expanded_target)
                        else:
                            path = os.path.expanduser(expanded_target)
                        if os.path.exists(path):
                            expanded_paths.append(path)

                if expanded_paths:
                    # Multiple files - use existing file gathering logic
                    refs = create_file_references(
                        expanded_paths, format=output_format, label=label_style
                    )
                    return refs["concatenated"]
                else:
                    raise Exception(f"No files found matching pattern: {target}")

            # Single path handling
            if os.path.isfile(full_target):
                # Single file
                try:
                    with open(full_target, "r", encoding="utf-8") as f:
                        content = f.read()
                    if label_style == "relative":
                        label = os.path.relpath(full_target)
                    elif label_style == "name":
                        label = os.path.basename(full_target)
                    elif label_style == "ext":
                        label = os.path.splitext(full_target)[1]
                    else:
                        label = full_target
                    from .reference import process_text

                    return process_text(content, format=output_format, label=label)
                except UnicodeDecodeError:
                    raise Exception(
                        f"File is not UTF-8 text: {full_target} (appears to be binary or use non-UTF-8 encoding)"
                    )
            elif os.path.isdir(full_target):
                # Directory
                refs = create_file_references(
                    [full_target], format=output_format, label=label_style
                )
                return refs["concatenated"]
            else:
                raise Exception(f"Path not found: {full_target}")

        except PermissionError:
            raise Exception(f"Permission denied accessing: {target}")
        except Exception as e:
            # Re-raise our own exceptions, wrap others
            if (
                "No files found matching pattern" in str(e)
                or "Path not found" in str(e)
                or "Permission denied" in str(e)
                or "File is not UTF-8 text" in str(e)
            ):
                raise e
            else:
                raise Exception(f"Error fetching local content {target}: {str(e)}")

    def process_injection(self, injection_data: Dict[str, Union[str, None]]) -> str:
        """Process a single injection and return the replacement content."""
        target = injection_data["target"]
        filename = injection_data["filename"]
        params = injection_data["params"]
        root = injection_data["root"]

        if self.is_http_url(target):
            return self.fetch_http_content(target, filename)
        elif self.is_git_target(target):
            return self.fetch_git_content(target, params)
        else:
            return self.fetch_local_content(target, root, params)

    def inject_content(self, text: str, max_depth: int = 5) -> str:
        """
        Process all {cx::...} patterns in text, recursively handling nested injections.

        Args:
            text: Input text containing injection patterns
            max_depth: Maximum recursion depth to prevent infinite loops

        Returns:
            Text with all injection patterns replaced with their content

        Raises:
            Exception: If any injection fails to process
        """
        if max_depth <= 0:
            return text

        # Find all injection patterns
        matches = list(self.INJECTION_PATTERN.finditer(text))
        if not matches:
            return text

        # Process matches in reverse order to maintain string positions
        result = text
        processed_count = 0

        for match in reversed(matches):
            injection_data = self.parse_injection(match)

            # Validate that we have a target
            if not injection_data.get("target"):
                raise Exception(f"Empty injection target in pattern: {match.group(0)}")

            # Process the injection - let any exceptions propagate
            replacement_content = self.process_injection(injection_data)
            processed_count += 1

            # Replace the injection pattern with the fetched content
            start, end = match.span()
            result = result[:start] + replacement_content + result[end:]

        # Recursively process any new injection patterns in the replacement content
        return self.inject_content(result, max_depth - 1)


def inject_content_in_text(text: str) -> str:
    """
    Convenience function to inject content into text.

    Args:
        text: Input text containing {cx::...} patterns

    Returns:
        Text with all injection patterns replaced
    """
    parser = InjectionParser()
    return parser.inject_content(text)

