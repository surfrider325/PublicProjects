"""Authentication strategies applied to outgoing requests."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import (
    ApiKeyAuth,
    AuthConfig,
    BasicAuth,
    BearerAuth,
    NoAuth,
    OAuth2ClientCredentials,
)


class AuthError(Exception):
    """Raised when credentials cannot be obtained or are rejected."""


@dataclass
class Authenticator:
    """Mutates request headers/params in place; refreshes OAuth tokens as needed."""

    config: AuthConfig
    _token: str | None = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)

    def apply(
        self,
        client: httpx.Client | None,
        headers: dict[str, str],
        params: dict[str, Any],
    ) -> None:
        """Add credentials to the outgoing request.

        `client` is only used by OAuth2, which must call the token endpoint;
        every other strategy ignores it.
        """
        cfg = self.config
        if isinstance(cfg, NoAuth):
            return
        if isinstance(cfg, BearerAuth):
            headers[cfg.header] = f"{cfg.prefix} {cfg.token}".strip()
        elif isinstance(cfg, ApiKeyAuth):
            if cfg.location == "header":
                headers[cfg.name] = cfg.key
            else:
                params[cfg.name] = cfg.key
        elif isinstance(cfg, BasicAuth):
            raw = f"{cfg.username}:{cfg.password}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        elif isinstance(cfg, OAuth2ClientCredentials):
            if client is None:
                raise AuthError("oauth2 auth requires an HTTP client")
            headers["Authorization"] = f"Bearer {self._oauth_token(client, cfg)}"
        else:  # pragma: no cover - guarded by the config union
            raise AuthError(f"unsupported auth type: {cfg!r}")

    def invalidate(self) -> None:
        """Drop any cached token so the next request re-authenticates."""
        self._token = None
        self._expires_at = 0.0

    def _oauth_token(self, client: httpx.Client, cfg: OAuth2ClientCredentials) -> str:
        # 60s safety margin so a token can't expire mid-flight.
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        data: dict[str, str] = {"grant_type": "client_credentials"}
        headers: dict[str, str] = {}
        if cfg.send_credentials_in == "body":
            data["client_id"] = cfg.client_id
            data["client_secret"] = cfg.client_secret
        else:
            raw = f"{cfg.client_id}:{cfg.client_secret}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        if cfg.scope:
            data["scope"] = cfg.scope
        if cfg.audience:
            data["audience"] = cfg.audience

        response = client.post(cfg.token_url, data=data, headers=headers, timeout=30.0)
        if response.status_code >= 400:
            raise AuthError(
                f"token request to {cfg.token_url} failed "
                f"({response.status_code}): {response.text[:300]}"
            )
        payload = response.json()
        token = payload.get("access_token")
        if not token:
            raise AuthError(f"token response from {cfg.token_url} has no access_token")
        self._token = token
        self._expires_at = time.time() + float(payload.get("expires_in", 3600))
        return token
