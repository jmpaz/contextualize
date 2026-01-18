from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..render.text import process_text
from ..utils import count_tokens

_YOUTUBE_URL_RE = re.compile(
    r"^https?://(?:www\.|m\.)?(?:"
    r"youtube\.com/(?:watch\?.*v=|shorts/|live/)|"
    r"youtu\.be/"
    r")(?P<id>[\w-]{11})"
)


def is_youtube_url(url: str) -> bool:
    return bool(_YOUTUBE_URL_RE.match(url))


def extract_video_id(url: str) -> str | None:
    match = _YOUTUBE_URL_RE.match(url)
    if match:
        return match.group("id")
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        params = parse_qs(parsed.query)
        v = params.get("v")
        if v:
            return v[0]
    return None


def _check_ytdlp() -> None:
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "YouTube processing requires yt-dlp: https://github.com/yt-dlp/yt-dlp"
        )


def _get_timestamp_interval(duration_seconds: int) -> int:
    if duration_seconds < 300:
        return 60
    if duration_seconds < 1800:
        return 180
    if duration_seconds < 3600:
        return 300
    return 600


def _format_timestamp(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"[{hours}:{minutes:02d}:{secs:02d}]"
    return f"[{minutes}:{secs:02d}]"


def _insert_timestamps(text: str, duration_seconds: int) -> str:
    if duration_seconds <= 0:
        return text

    interval = _get_timestamp_interval(duration_seconds)
    paragraphs = text.split("\n\n")
    if len(paragraphs) <= 1:
        return text

    total_chars = sum(len(p) for p in paragraphs)
    if total_chars == 0:
        return text

    result = []
    cumulative_chars = 0

    for i, paragraph in enumerate(paragraphs):
        position_ratio = cumulative_chars / total_chars
        current_time = int(position_ratio * duration_seconds)
        marker_time = (current_time // interval) * interval

        if i > 0 and marker_time > 0:
            prev_ratio = (cumulative_chars - len(paragraphs[i - 1])) / total_chars
            prev_time = int(prev_ratio * duration_seconds)
            prev_marker = (prev_time // interval) * interval
            if marker_time > prev_marker:
                paragraph = f"{_format_timestamp(marker_time)}\n{paragraph}"

        result.append(paragraph)
        cumulative_chars += len(paragraph)

    return "\n\n".join(result)


def _split_into_paragraphs(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    paragraphs = []
    current = []
    for sent in sentences:
        current.append(sent)
        if len(current) >= 4:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs)


def _escape_yaml_string(s: str) -> str:
    if not s:
        return '""'
    if "\n" in s:
        indented = "\n".join("  " + line for line in s.split("\n"))
        return f"|\n{indented}"
    needs_quotes = any(c in s for c in ":{}[],\"'|>&*!?#%@`") or s.startswith(" ")
    if needs_quotes:
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


@dataclass
class YouTubeReference:
    url: str
    format: str = "md"
    label: str = "relative"
    token_target: str = "cl100k_base"
    include_token_count: bool = False
    label_suffix: str | None = None
    inject: bool = False
    depth: int = 5
    trace_collector: list = None
    use_cache: bool = True
    cache_ttl: timedelta | None = None
    refresh_cache: bool = False
    _video_id: str | None = field(default=None, init=False, repr=False)
    _metadata: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        _check_ytdlp()
        self._video_id = extract_video_id(self.url)
        if not self._video_id:
            raise ValueError(f"Could not extract video ID from URL: {self.url}")
        self.file_content = ""
        self.original_file_content = ""
        self.output = self._get_contents()

    @property
    def path(self) -> str:
        return self.url

    def read(self) -> str:
        return self.original_file_content

    def exists(self) -> bool:
        try:
            self._fetch_metadata()
            return True
        except Exception:
            return False

    def token_count(self, encoding: str = "cl100k_base") -> int:
        return count_tokens(self.original_file_content, target=encoding)["count"]

    def get_label(self) -> str:
        if self.label == "relative":
            return self.url
        if self.label == "name":
            return self._video_id or self.url
        if self.label == "ext":
            return ""
        return self.label

    def _fetch_metadata(self) -> dict:
        if self._metadata is not None:
            return self._metadata

        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", self.url],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp metadata failed: {result.stderr}")

        self._metadata = json.loads(result.stdout)
        return self._metadata

    def _fetch_captions(self) -> str | None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "%(id)s")
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang",
                    "en",
                    "--sub-format",
                    "vtt",
                    "--skip-download",
                    "-o",
                    output_template,
                    self.url,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return None

            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                return None

            vtt_content = vtt_files[0].read_text(encoding="utf-8")
            return _parse_vtt(vtt_content)

    def _extract_audio(self) -> Path:
        tmpdir = tempfile.mkdtemp()
        output_path = os.path.join(tmpdir, f"{self._video_id}.mp3")

        result = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format",
                "mp3",
                "--audio-quality",
                "5",
                "-o",
                output_path,
                self.url,
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp audio extraction failed: {result.stderr}")

        return Path(output_path)

    def _transcribe_audio(self, audio_path: Path) -> str:
        import httpx

        api_base = os.environ.get("WHISPER_API_BASE", "https://api.openai.com/v1")
        api_key = os.environ.get("WHISPER_API_KEY")
        model = os.environ.get("WHISPER_MODEL", "whisper-1")
        endpoint = f"{api_base.rstrip('/')}/audio/transcriptions"

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with open(audio_path, "rb") as f:
            response = httpx.post(
                endpoint,
                headers=headers,
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data={"model": model, "response_format": "verbose_json"},
                timeout=600,
            )

        if response.status_code != 200:
            raise RuntimeError(
                f"Whisper API error: {response.status_code} {response.text}"
            )

        data = response.json()
        segments = data.get("segments", [])
        if segments:
            return "\n\n".join(seg.get("text", "").strip() for seg in segments)
        return data.get("text", "").strip()

    def _get_transcript(self, duration: int) -> tuple[str, str]:
        whisper_configured = bool(os.environ.get("WHISPER_API_BASE"))

        if whisper_configured:
            audio_path = None
            try:
                audio_path = self._extract_audio()
                transcript = self._transcribe_audio(audio_path)
                return transcript, "whisper"
            finally:
                if audio_path and audio_path.exists():
                    shutil.rmtree(audio_path.parent, ignore_errors=True)
        else:
            captions = self._fetch_captions()
            if captions:
                return captions, "captions"
            return "", "none"

    def _format_output(self, metadata: dict, transcript: str, source: str) -> str:
        title = metadata.get("title", "Untitled")
        channel = metadata.get("channel") or metadata.get("uploader")
        description = metadata.get("description")
        duration = metadata.get("duration", 0)

        has_rich_metadata = channel or description

        lines = []

        if has_rich_metadata:
            lines.append("---")
            lines.append(f"title: {_escape_yaml_string(title)}")
            if channel:
                lines.append(f"channel: {_escape_yaml_string(channel)}")
            if description:
                lines.append(f"description: {_escape_yaml_string(description)}")
            lines.append("---")
            lines.append("")
        else:
            lines.append(f"# {title}")
            lines.append("")

        if transcript:
            formatted_transcript = _insert_timestamps(transcript, duration)
            lines.append(formatted_transcript)
        elif source == "none":
            lines.append("*No transcript available.*")

        return "\n".join(lines)

    def _get_contents(self) -> str:
        whisper_available = bool(os.environ.get("WHISPER_API_BASE"))

        if self.use_cache and not self.refresh_cache:
            from ..cache.youtube import get_cached_transcript

            cached = get_cached_transcript(
                self._video_id, self.cache_ttl, whisper_available=whisper_available
            )
            if cached is not None:
                self.original_file_content = cached
                self.file_content = cached
                text = cached
                if self.inject:
                    from ..render.inject import inject_content_in_text

                    text = inject_content_in_text(
                        text, self.depth, self.trace_collector, self.url
                    )
                    self.file_content = text
                return process_text(
                    text,
                    format=self.format,
                    label=self.get_label(),
                    label_suffix=self.label_suffix,
                    token_target=self.token_target,
                    include_token_count=self.include_token_count,
                )

        metadata = self._fetch_metadata()
        duration = metadata.get("duration", 0)
        transcript, source = self._get_transcript(duration)
        text = self._format_output(metadata, transcript, source)

        self.original_file_content = text
        self.file_content = text

        if self.use_cache:
            from ..cache.youtube import store_transcript

            store_transcript(self._video_id, text, source)

        if self.inject:
            from ..render.inject import inject_content_in_text

            text = inject_content_in_text(
                text, self.depth, self.trace_collector, self.url
            )
            self.file_content = text

        return process_text(
            text,
            format=self.format,
            label=self.get_label(),
            label_suffix=self.label_suffix,
            token_target=self.token_target,
            include_token_count=self.include_token_count,
        )

    def get_contents(self) -> str:
        return self.output


def _parse_vtt(vtt_content: str) -> str:
    lines = vtt_content.split("\n")
    text_lines = []
    seen = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT"):
            continue
        if line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(
            r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}", line
        ):
            continue
        if line.startswith("NOTE"):
            continue

        clean = re.sub(r"<[^>]+>", "", line)
        clean = re.sub(r"&nbsp;", " ", clean)
        clean = clean.strip()

        if clean and clean not in seen:
            seen.add(clean)
            text_lines.append(clean)

    text = " ".join(text_lines)
    return _split_into_paragraphs(text)
