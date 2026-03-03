from __future__ import annotations

import base64
import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

_DEFAULT_OAUTH_BASE = "https://secure.soundcloud.com"
_DEFAULT_CALLBACK_HOST = "127.0.0.1"
_DEFAULT_CALLBACK_PORT = 8976
_DEFAULT_CALLBACK_PATH = "/soundcloud/callback"


@dataclass(frozen=True)
class SoundCloudOAuthConfig:
    client_id: str
    client_secret: str
    authorize_url: str
    token_url: str
    callback_host: str
    callback_port: int
    callback_path: str
    redirect_uri: str


@dataclass(frozen=True)
class SoundCloudClientCredentials:
    client_id: str
    client_secret: str
    token_url: str


@dataclass(frozen=True)
class SoundCloudToken:
    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str
    expires_in: int
    issued_at: datetime

    @property
    def expires_at(self) -> datetime:
        return self.issued_at + timedelta_seconds(self.expires_in)

    def as_cache_payload(self) -> dict[str, str | int]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token or "",
            "token_type": self.token_type,
            "scope": self.scope,
            "expires_in": self.expires_in,
            "issued_at": self.issued_at.astimezone(timezone.utc).isoformat(),
            "expires_at": self.expires_at.astimezone(timezone.utc).isoformat(),
        }


def timedelta_seconds(seconds: int):
    from datetime import timedelta

    return timedelta(seconds=max(0, int(seconds)))


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv

        env_path = find_dotenv(usecwd=True)
        if env_path:
            load_dotenv(env_path, override=False)
    except Exception:
        return


def _oauth_base() -> str:
    return (os.environ.get("SOUNDCLOUD_OAUTH_BASE") or _DEFAULT_OAUTH_BASE).strip()


def _callback_host() -> str:
    value = (os.environ.get("SOUNDCLOUD_OAUTH_CALLBACK_HOST") or "").strip()
    return value or _DEFAULT_CALLBACK_HOST


def _callback_port() -> int:
    raw = (os.environ.get("SOUNDCLOUD_OAUTH_CALLBACK_PORT") or "").strip()
    if not raw:
        return _DEFAULT_CALLBACK_PORT
    try:
        parsed = int(raw)
    except ValueError:
        return _DEFAULT_CALLBACK_PORT
    if parsed < 1 or parsed > 65535:
        return _DEFAULT_CALLBACK_PORT
    return parsed


def _callback_path() -> str:
    raw = (os.environ.get("SOUNDCLOUD_OAUTH_CALLBACK_PATH") or "").strip()
    if not raw:
        return _DEFAULT_CALLBACK_PATH
    if raw.startswith("/"):
        return raw
    return "/" + raw


def _default_redirect_uri() -> str:
    return f"http://{_callback_host()}:{_callback_port()}{_callback_path()}"


def load_soundcloud_client_credentials() -> SoundCloudClientCredentials:
    _load_dotenv()
    client_id = (os.environ.get("SOUNDCLOUD_CLIENT_ID") or "").strip()
    client_secret = (os.environ.get("SOUNDCLOUD_CLIENT_SECRET") or "").strip()
    if not client_id or not client_secret:
        raise ValueError(
            "SOUNDCLOUD_CLIENT_ID and SOUNDCLOUD_CLIENT_SECRET are required"
        )
    oauth_base = _oauth_base().rstrip("/")
    return SoundCloudClientCredentials(
        client_id=client_id,
        client_secret=client_secret,
        token_url=f"{oauth_base}/oauth/token",
    )


def load_soundcloud_oauth_config() -> SoundCloudOAuthConfig:
    creds = load_soundcloud_client_credentials()
    oauth_base = _oauth_base().rstrip("/")

    redirect_uri = (
        os.environ.get("SOUNDCLOUD_OAUTH_REDIRECT_URI") or _default_redirect_uri()
    ).strip()
    parsed = urlparse(redirect_uri)
    if parsed.scheme.lower() != "http" or not parsed.hostname or not parsed.port:
        raise ValueError(
            "SOUNDCLOUD_OAUTH_REDIRECT_URI must be an http:// loopback URL with an explicit port"
        )
    callback_path = parsed.path or "/"
    if not callback_path.startswith("/"):
        callback_path = "/" + callback_path

    return SoundCloudOAuthConfig(
        client_id=creds.client_id,
        client_secret=creds.client_secret,
        authorize_url=f"{oauth_base}/authorize",
        token_url=creds.token_url,
        callback_host=parsed.hostname,
        callback_port=int(parsed.port),
        callback_path=callback_path,
        redirect_uri=redirect_uri,
    )


def generate_pkce_verifier(length: int = 64) -> str:
    if length < 43:
        length = 43
    if length > 128:
        length = 128
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def build_pkce_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def build_authorize_url(
    config: SoundCloudOAuthConfig,
    *,
    code_challenge: str,
    state: str,
) -> str:
    query = urlencode(
        {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "response_type": "code",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    return f"{config.authorize_url}?{query}"


def exchange_authorization_code(
    config: SoundCloudOAuthConfig,
    *,
    code: str,
    code_verifier: str,
) -> SoundCloudToken:
    import requests

    response = requests.post(
        config.token_url,
        headers={
            "accept": "application/json; charset=utf-8",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "redirect_uri": config.redirect_uri,
            "code_verifier": code_verifier,
            "code": code,
        },
        timeout=30,
    )
    response.raise_for_status()
    return parse_token_response(response.json())


def refresh_access_token(
    config: SoundCloudOAuthConfig | SoundCloudClientCredentials,
    *,
    refresh_token: str,
) -> SoundCloudToken:
    import requests

    response = requests.post(
        config.token_url,
        headers={
            "accept": "application/json; charset=utf-8",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    response.raise_for_status()
    return parse_token_response(response.json())


def parse_token_response(payload: dict[str, object]) -> SoundCloudToken:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError("SoundCloud token response did not include an access_token")

    refresh_token = payload.get("refresh_token")
    normalized_refresh = (
        refresh_token.strip()
        if isinstance(refresh_token, str) and refresh_token.strip()
        else None
    )
    token_type = payload.get("token_type")
    normalized_token_type = (
        token_type.strip()
        if isinstance(token_type, str) and token_type.strip()
        else "Bearer"
    )
    scope = payload.get("scope")
    normalized_scope = scope.strip() if isinstance(scope, str) else ""
    expires_in = payload.get("expires_in")
    if not isinstance(expires_in, int):
        try:
            expires_in = int(str(expires_in))
        except (TypeError, ValueError):
            expires_in = 3600
    expires_in = max(60, expires_in)
    return SoundCloudToken(
        access_token=access_token.strip(),
        refresh_token=normalized_refresh,
        token_type=normalized_token_type,
        scope=normalized_scope,
        expires_in=expires_in,
        issued_at=datetime.now(timezone.utc),
    )


@dataclass(frozen=True)
class SoundCloudOAuthCallbackResult:
    code: str
    state: str


def wait_for_oauth_callback(
    config: SoundCloudOAuthConfig,
    *,
    expected_state: str,
    timeout_seconds: int = 180,
) -> SoundCloudOAuthCallbackResult:
    received: dict[str, str] = {}
    event = threading.Event()

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _respond(self, code: int, body: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != config.callback_path:
                self._respond(404, "<html><body><h1>Not Found</h1></body></html>")
                return
            query = dict(parse_qs(parsed.query))
            state = (query.get("state", [""])[0] or "").strip()
            code = (query.get("code", [""])[0] or "").strip()
            error = (query.get("error", [""])[0] or "").strip()
            if error:
                received["error"] = error
                event.set()
                self._respond(
                    200,
                    "<html><body><h1>SoundCloud sign-in failed.</h1><p>You can close this window.</p></body></html>",
                )
                return
            if not code:
                self._respond(
                    400,
                    "<html><body><h1>Missing authorization code.</h1><p>You can close this window.</p></body></html>",
                )
                return
            if not state or state != expected_state:
                received["error"] = "state_mismatch"
                event.set()
                self._respond(
                    400,
                    "<html><body><h1>State mismatch.</h1><p>You can close this window.</p></body></html>",
                )
                return
            received["code"] = code
            received["state"] = state
            event.set()
            self._respond(
                200,
                "<html><body><h1>SoundCloud sign-in complete.</h1><p>You can close this window.</p></body></html>",
            )

    server = ThreadingHTTPServer(
        (config.callback_host, config.callback_port),
        CallbackHandler,
    )
    server.timeout = 0.5

    try:
        deadline = time.monotonic() + max(5, timeout_seconds)
        while time.monotonic() < deadline and not event.is_set():
            server.handle_request()
    finally:
        server.server_close()

    error = received.get("error")
    if error:
        raise ValueError(f"SoundCloud OAuth callback failed: {error}")

    code = (received.get("code") or "").strip()
    state = (received.get("state") or "").strip()
    if not code or not state:
        raise TimeoutError(
            "Timed out waiting for SoundCloud OAuth callback. Ensure your app redirect URI matches this callback URL."
        )
    return SoundCloudOAuthCallbackResult(code=code, state=state)
