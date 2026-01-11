"""Manifest dataclass definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FileSpec:
    """Specification for a file in a component."""

    path: str
    """Path, URL, or git target."""

    options: dict[str, Any] = field(default_factory=dict)
    """Additional options like filename, wrap, link-depth, etc."""

    @property
    def is_url(self) -> bool:
        """Check if this is a URL spec."""
        return self.path.startswith("http://") or self.path.startswith("https://")

    @property
    def is_git_target(self) -> bool:
        """Check if this is a git target spec."""
        for prefix in ("github:", "gh:", "git@"):
            if self.path.startswith(prefix):
                return True
        if self.path.startswith("http") and ".git" in self.path:
            return True
        return False


@dataclass
class Component:
    """A component in a manifest."""

    name: str
    """Unique name for this component."""

    files: list[FileSpec] = field(default_factory=list)
    """List of file specifications."""

    text: str | None = None
    """Inline text content (alternative to files)."""

    prefix: str | None = None
    """Content to prepend to the component output."""

    suffix: str | None = None
    """Content to append to the component output."""

    wrap: str | None = None
    """Wrapping mode (md, xml, etc.)."""

    comment: str | None = None
    """Metadata comment for the component."""

    link_depth: int = 0
    """Depth for markdown link traversal."""

    link_scope: str = "all"
    """Scope for link traversal (all or first)."""

    link_skip: list[str] = field(default_factory=list)
    """Paths to skip when resolving markdown links."""

    group_path: tuple[str, ...] | None = None
    """Path of parent groups (for nested components)."""

    group_base: str | None = None
    """Base name within the innermost group."""

    def __post_init__(self):
        """Normalize fields after initialization."""
        if self.link_skip is None:
            self.link_skip = []

    @property
    def is_text_only(self) -> bool:
        """Check if this is a text-only component (no files)."""
        return self.text is not None and not self.files


@dataclass
class ContextConfig:
    """Configuration for context/hydration output."""

    dir: str = ".context"
    """Output directory for hydration."""

    access: str = "writable"
    """Access mode: 'writable' or 'read-only'."""

    path_strategy: str = "on-disk"
    """Path strategy: 'on-disk' or 'by-component'."""

    include_meta: bool = True
    """Whether to emit manifest.yaml and index.json."""

    agents: dict[str, Any] = field(default_factory=dict)
    """Agent configuration (text, files, etc.)."""


@dataclass
class ManifestConfig:
    """Top-level manifest configuration."""

    root: str | None = None
    """Base directory for relative paths."""

    link_depth: int = 0
    """Default link traversal depth."""

    link_scope: str = "all"
    """Default link scope."""

    link_skip: list[str] = field(default_factory=list)
    """Default paths to skip."""

    context: ContextConfig = field(default_factory=ContextConfig)
    """Context/hydration configuration."""


@dataclass
class Manifest:
    """A parsed manifest structure."""

    config: ManifestConfig
    """Top-level configuration."""

    components: list[Component]
    """List of components."""

    groups: dict[str, list[str]] = field(default_factory=dict)
    """Mapping of group names to component names."""

    raw_data: dict[str, Any] = field(default_factory=dict)
    """Original raw manifest data for reference."""

    @property
    def component_names(self) -> list[str]:
        """Get all component names."""
        return [c.name for c in self.components]

    def get_component(self, name: str) -> Component | None:
        """Get a component by name."""
        for c in self.components:
            if c.name == name:
                return c
        return None

    def get_components_in_group(self, group: str) -> list[Component]:
        """Get all components in a group."""
        component_names = self.groups.get(group, [])
        return [c for c in self.components if c.name in component_names]
