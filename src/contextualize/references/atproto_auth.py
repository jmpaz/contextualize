from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse, urlsplit, urlunsplit

import jwt

_DEFAULT_RESOURCE_SERVER = "https://bsky.social"
_DEFAULT_CALLBACK_HOST = "127.0.0.1"
_DEFAULT_CALLBACK_PORT = 8978
_DEFAULT_CALLBACK_PATH = "/atproto/callback"
_DEFAULT_SCOPE = "atproto transition:generic"


@dataclass(frozen=True)
class AtprotoOAuthConfig:
    client_id: str
    scope: str
    redirect_uri: str
    callback_host: str
    callback_port: int
    callback_path: str
    resource_server: str
    authorization_server: str
    authorization_endpoint: str
    token_endpoint: str
    par_endpoint: str


@dataclass(frozen=True)
class AtprotoOAuthToken:
    access_token: str
    refresh_token: str | None
    token_type: str
    scope: str
    expires_in: int
    issued_at: datetime
    subject_did: str | None = None

    @property
    def expires_at(self) -> datetime:
        return self.issued_at + timedelta_seconds(self.expires_in)


@dataclass(frozen=True)
class AtprotoOAuthCallbackResult:
    code: str
    state: str


@dataclass(frozen=True)
class DPoPKeyPair:
    private_key_pem: str
    public_jwk: dict[str, str]


class AtprotoDPoPAuth:
    def __init__(
        self,
        *,
        access_token: str,
        private_key_pem: str,
        public_jwk: dict[str, str],
        token_type: str = "DPoP",
        resource_nonce: str | None = None,
    ) -> None:
        self._token = access_token.strip()
        self._private_key_pem = private_key_pem
        self._public_jwk = public_jwk
        self._token_type = token_type or "DPoP"
        self._resource_nonce = resource_nonce

    @property
    def resource_nonce(self) -> str | None:
        return self._resource_nonce

    @property
    def token(self) -> str:
        return self._token

    def update_token(self, access_token: str) -> None:
        cleaned = access_token.strip()
        if cleaned:
            self._token = cleaned

    def update_nonce(self, nonce: str | None) -> None:
        cleaned = (nonce or "").strip()
        self._resource_nonce = cleaned or None

    def __call__(self, request):
        ath = _access_token_hash(self.token)
        proof = _build_dpop_proof(
            private_key_pem=self._private_key_pem,
            public_jwk=self._public_jwk,
            http_method=request.method,
            http_url=request.url,
            nonce=self._resource_nonce,
            ath=ath,
        )
        request.headers["Authorization"] = f"{self._token_type} {self.token}"
        request.headers["DPoP"] = proof
        request.register_hook("response", self._capture_nonce)
        return request

    def _capture_nonce(self, response, *args, **kwargs):
        nonce = (response.headers.get("DPoP-Nonce") or "").strip()
        if nonce:
            self._resource_nonce = nonce
        return response


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


def _callback_host() -> str:
    value = (os.environ.get("ATPROTO_OAUTH_CALLBACK_HOST") or "").strip()
    return value or _DEFAULT_CALLBACK_HOST


def _callback_port() -> int:
    raw = (os.environ.get("ATPROTO_OAUTH_CALLBACK_PORT") or "").strip()
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
    raw = (os.environ.get("ATPROTO_OAUTH_CALLBACK_PATH") or "").strip()
    if not raw:
        return _DEFAULT_CALLBACK_PATH
    if raw.startswith("/"):
        return raw
    return "/" + raw


def _default_redirect_uri() -> str:
    return f"http://{_callback_host()}:{_callback_port()}{_callback_path()}"


def _base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _decode_unverified_jwt_sub(token: str) -> str | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload_part = parts[1]
    padding = "=" * (-len(payload_part) % 4)
    try:
        decoded = base64.urlsafe_b64decode((payload_part + padding).encode("ascii"))
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    sub = payload.get("sub")
    if not isinstance(sub, str):
        return None
    cleaned = sub.strip()
    return cleaned or None


def _normalize_origin(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower() or "https"
    host = parsed.hostname or ""
    port = parsed.port
    default_port = (scheme == "https" and port == 443) or (
        scheme == "http" and port == 80
    )
    netloc = host if (port is None or default_port) else f"{host}:{port}"
    return f"{scheme}://{netloc}"


def _normalize_htu(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", "", ""))


def _default_client_id(redirect_uri: str, scope: str) -> str:
    query = urlencode({"redirect_uri": redirect_uri, "scope": scope})
    return f"http://localhost/?{query}"


def _discover_authorization_server(resource_server: str) -> str:
    import requests

    resource_origin = _normalize_origin(resource_server)
    response = requests.get(
        f"{resource_origin}/.well-known/oauth-protected-resource",
        headers={"Accept": "application/json"},
        timeout=20,
    )
    if response.status_code == 404:
        return resource_origin
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("ATProto resource metadata must be an object")
    servers = payload.get("authorization_servers")
    if isinstance(servers, list):
        for item in servers:
            if isinstance(item, str) and item.strip():
                return _normalize_origin(item.strip())
    issuer = payload.get("issuer")
    if isinstance(issuer, str) and issuer.strip():
        return _normalize_origin(issuer.strip())
    return resource_origin


def _discover_authorization_metadata(authorization_server: str) -> dict[str, object]:
    import requests

    response = requests.get(
        f"{_normalize_origin(authorization_server)}/.well-known/oauth-authorization-server",
        headers={"Accept": "application/json"},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("ATProto authorization metadata must be an object")
    return payload


def load_atproto_oauth_config() -> AtprotoOAuthConfig:
    _load_dotenv()
    redirect_uri = (
        os.environ.get("ATPROTO_OAUTH_REDIRECT_URI") or _default_redirect_uri()
    ).strip()
    parsed = urlparse(redirect_uri)
    if parsed.scheme.lower() != "http" or not parsed.hostname or not parsed.port:
        raise ValueError(
            "ATPROTO_OAUTH_REDIRECT_URI must be an http:// loopback URL with an explicit port"
        )
    callback_path = parsed.path or "/"
    if not callback_path.startswith("/"):
        callback_path = "/" + callback_path

    scope = (
        os.environ.get("ATPROTO_OAUTH_SCOPE") or _DEFAULT_SCOPE
    ).strip() or _DEFAULT_SCOPE
    resource_server = (
        os.environ.get("ATPROTO_OAUTH_RESOURCE_SERVER")
        or os.environ.get("ATPROTO_PDS_HOST")
        or _DEFAULT_RESOURCE_SERVER
    ).strip()
    resource_server = _normalize_origin(resource_server)

    auth_server = _discover_authorization_server(resource_server)
    metadata = _discover_authorization_metadata(auth_server)

    authorization_endpoint = metadata.get("authorization_endpoint")
    token_endpoint = metadata.get("token_endpoint")
    par_endpoint = metadata.get("pushed_authorization_request_endpoint")
    if (
        not isinstance(authorization_endpoint, str)
        or not authorization_endpoint.strip()
    ):
        raise ValueError(
            "ATProto authorization metadata is missing authorization_endpoint"
        )
    if not isinstance(token_endpoint, str) or not token_endpoint.strip():
        raise ValueError("ATProto authorization metadata is missing token_endpoint")
    if not isinstance(par_endpoint, str) or not par_endpoint.strip():
        raise ValueError(
            "ATProto authorization metadata is missing pushed_authorization_request_endpoint"
        )

    client_id = (os.environ.get("ATPROTO_OAUTH_CLIENT_ID") or "").strip()
    if not client_id:
        client_id = _default_client_id(redirect_uri, scope)

    return AtprotoOAuthConfig(
        client_id=client_id,
        scope=scope,
        redirect_uri=redirect_uri,
        callback_host=parsed.hostname,
        callback_port=int(parsed.port),
        callback_path=callback_path,
        resource_server=resource_server,
        authorization_server=_normalize_origin(auth_server),
        authorization_endpoint=authorization_endpoint.strip(),
        token_endpoint=token_endpoint.strip(),
        par_endpoint=par_endpoint.strip(),
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
    return _base64url(digest)


def generate_state() -> str:
    return secrets.token_urlsafe(24)


def generate_dpop_key_pair() -> DPoPKeyPair:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    public_numbers = private_key.public_key().public_numbers()
    x = public_numbers.x.to_bytes(32, "big")
    y = public_numbers.y.to_bytes(32, "big")
    return DPoPKeyPair(
        private_key_pem=private_key_pem,
        public_jwk={
            "kty": "EC",
            "crv": "P-256",
            "x": _base64url(x),
            "y": _base64url(y),
        },
    )


def _access_token_hash(access_token: str) -> str:
    digest = hashlib.sha256(access_token.encode("ascii")).digest()
    return _base64url(digest)


def _build_dpop_proof(
    *,
    private_key_pem: str,
    public_jwk: dict[str, str],
    http_method: str,
    http_url: str,
    nonce: str | None = None,
    ath: str | None = None,
) -> str:
    now = int(time.time())
    claims: dict[str, object] = {
        "jti": secrets.token_urlsafe(16),
        "htm": http_method.upper(),
        "htu": _normalize_htu(http_url),
        "iat": now,
    }
    if nonce:
        claims["nonce"] = nonce
    if ath:
        claims["ath"] = ath
    headers = {
        "typ": "dpop+jwt",
        "alg": "ES256",
        "jwk": public_jwk,
    }
    encoded = jwt.encode(claims, private_key_pem, algorithm="ES256", headers=headers)
    if not isinstance(encoded, str):
        raise ValueError("Failed to generate DPoP proof")
    return encoded


def _oauth_post_with_dpop(
    *,
    url: str,
    form: dict[str, str],
    private_key_pem: str,
    public_jwk: dict[str, str],
    nonce: str | None,
) -> tuple[dict[str, object], str | None]:
    import requests

    def _response_error_detail(response: requests.Response) -> str:
        detail = ""
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            description = payload.get("error_description")
            pieces = [
                str(part).strip()
                for part in (error, description)
                if isinstance(part, str) and part.strip()
            ]
            if pieces:
                detail = ": ".join(pieces)
        if not detail:
            text = response.text.strip()
            if text:
                detail = text[:500]
        return detail

    current_nonce = nonce
    for _ in range(2):
        dpop = _build_dpop_proof(
            private_key_pem=private_key_pem,
            public_jwk=public_jwk,
            http_method="POST",
            http_url=url,
            nonce=current_nonce,
        )
        response = requests.post(
            url,
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "DPoP": dpop,
            },
            data=form,
            timeout=30,
        )
        next_nonce = (response.headers.get("DPoP-Nonce") or "").strip() or None
        if response.ok:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("ATProto OAuth response must be an object")
            return payload, next_nonce
        if (
            response.status_code in {400, 401}
            and next_nonce
            and next_nonce != current_nonce
        ):
            current_nonce = next_nonce
            continue
        detail = _response_error_detail(response)
        if detail:
            raise ValueError(
                f"ATProto OAuth request failed for {url} ({response.status_code}): {detail}"
            )
        response.raise_for_status()
    raise RuntimeError("ATProto OAuth request failed after nonce retry")


def push_authorization_request(
    config: AtprotoOAuthConfig,
    *,
    code_challenge: str,
    state: str,
    key_pair: DPoPKeyPair,
    auth_server_nonce: str | None = None,
) -> tuple[str, str | None]:
    payload, next_nonce = _oauth_post_with_dpop(
        url=config.par_endpoint,
        form={
            "client_id": config.client_id,
            "response_type": "code",
            "redirect_uri": config.redirect_uri,
            "scope": config.scope,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        },
        private_key_pem=key_pair.private_key_pem,
        public_jwk=key_pair.public_jwk,
        nonce=auth_server_nonce,
    )
    request_uri = payload.get("request_uri")
    if not isinstance(request_uri, str) or not request_uri.strip():
        raise ValueError("ATProto PAR response did not include request_uri")
    return request_uri.strip(), next_nonce


def build_authorize_url(config: AtprotoOAuthConfig, *, request_uri: str) -> str:
    query = urlencode(
        {
            "client_id": config.client_id,
            "request_uri": request_uri,
        }
    )
    return f"{config.authorization_endpoint}?{query}"


def exchange_authorization_code(
    config: AtprotoOAuthConfig,
    *,
    code: str,
    code_verifier: str,
    key_pair: DPoPKeyPair,
    auth_server_nonce: str | None = None,
) -> tuple[AtprotoOAuthToken, str | None]:
    payload, next_nonce = _oauth_post_with_dpop(
        url=config.token_endpoint,
        form={
            "grant_type": "authorization_code",
            "client_id": config.client_id,
            "code": code,
            "redirect_uri": config.redirect_uri,
            "code_verifier": code_verifier,
        },
        private_key_pem=key_pair.private_key_pem,
        public_jwk=key_pair.public_jwk,
        nonce=auth_server_nonce,
    )
    return parse_token_response(payload), next_nonce


def refresh_access_token(
    config: AtprotoOAuthConfig,
    *,
    refresh_token: str,
    key_pair: DPoPKeyPair,
    auth_server_nonce: str | None = None,
) -> tuple[AtprotoOAuthToken, str | None]:
    payload, next_nonce = _oauth_post_with_dpop(
        url=config.token_endpoint,
        form={
            "grant_type": "refresh_token",
            "client_id": config.client_id,
            "refresh_token": refresh_token,
        },
        private_key_pem=key_pair.private_key_pem,
        public_jwk=key_pair.public_jwk,
        nonce=auth_server_nonce,
    )
    return parse_token_response(payload), next_nonce


def parse_token_response(payload: dict[str, object]) -> AtprotoOAuthToken:
    access_token = payload.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise ValueError("ATProto token response did not include an access_token")

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
        else "DPoP"
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
    subject_did = payload.get("sub")
    normalized_subject_did = (
        subject_did.strip()
        if isinstance(subject_did, str) and subject_did.strip()
        else None
    )
    if not normalized_subject_did:
        normalized_subject_did = _decode_unverified_jwt_sub(access_token.strip())
    return AtprotoOAuthToken(
        access_token=access_token.strip(),
        refresh_token=normalized_refresh,
        token_type=normalized_token_type,
        scope=normalized_scope,
        expires_in=expires_in,
        issued_at=datetime.now(timezone.utc),
        subject_did=normalized_subject_did,
    )


def wait_for_oauth_callback(
    config: AtprotoOAuthConfig,
    *,
    expected_state: str,
    timeout_seconds: int = 180,
) -> AtprotoOAuthCallbackResult:
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
                    "<html><body><h1>ATProto sign-in failed.</h1><p>You can close this window.</p></body></html>",
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
                "<html><body><h1>ATProto sign-in complete.</h1><p>You can close this window.</p></body></html>",
            )

    server = ThreadingHTTPServer(
        (config.callback_host, config.callback_port), CallbackHandler
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
        raise ValueError(f"ATProto OAuth callback failed: {error}")

    code = (received.get("code") or "").strip()
    state = (received.get("state") or "").strip()
    if not code or not state:
        raise TimeoutError(
            "Timed out waiting for ATProto OAuth callback. Ensure your app redirect URI matches this callback URL."
        )
    return AtprotoOAuthCallbackResult(code=code, state=state)


def extract_dpop_nonce_from_http_error(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    nonce = headers.get("DPoP-Nonce")
    if not isinstance(nonce, str):
        return None
    cleaned = nonce.strip()
    return cleaned or None
