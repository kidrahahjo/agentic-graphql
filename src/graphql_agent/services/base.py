"""Base abstractions for service clients."""

from __future__ import annotations

import abc
import logging
from collections.abc import Mapping
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)


class ServiceClient(abc.ABC):
    """Abstract service client that shares an HTTP session."""

    def __init__(self, base_url: str, auth_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=10.0)
        LOGGER.info(
            "Initialized ServiceClient base_url=%s auth_token_present=%s",
            self.base_url,
            bool(auth_token),
        )

    async def close(self) -> None:
        await self._client.aclose()
        LOGGER.debug("Closed HTTP client for %s", self.base_url)

    def _headers(self) -> Mapping[str, str]:
        headers: dict[str, str] = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        LOGGER.debug("Headers prepared for %s: %s", self.base_url, list(headers.keys()))
        return headers

    @abc.abstractmethod
    async def handle(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the service call and return JSON-like data."""


class JSONServiceClient(ServiceClient):
    """Convenience client for JSON APIs."""

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        full_url = f"{self.base_url}{path}"
        LOGGER.info("HTTP GET %s params=%s", full_url, params)
        response = await self._client.get(path, params=params, headers=self._headers())
        LOGGER.info(
            "HTTP GET %s -> status=%s content_length=%s",
            full_url,
            response.status_code,
            len(response.content),
        )
        LOGGER.debug(
            "HTTP GET %s response_body=%s", full_url, response.text[:500] if response.text else None
        )
        response.raise_for_status()
        parsed = response.json()
        LOGGER.debug("HTTP GET %s parsed_json=%s", full_url, parsed)
        return parsed

    async def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        full_url = f"{self.base_url}{path}"
        LOGGER.info(
            "HTTP POST %s params=%s payload_keys=%s",
            full_url,
            params,
            list(payload.keys()) if isinstance(payload, dict) else type(payload).__name__,
        )
        LOGGER.debug("HTTP POST %s payload=%s", full_url, payload)
        response = await self._client.post(
            path,
            json=payload,
            params=params,
            headers=self._headers(),
        )
        LOGGER.info(
            "HTTP POST %s -> status=%s content_length=%s",
            full_url,
            response.status_code,
            len(response.content),
        )
        LOGGER.debug(
            "HTTP POST %s response_body=%s",
            full_url,
            response.text[:500] if response.text else None,
        )
        response.raise_for_status()
        parsed = response.json()
        LOGGER.debug("HTTP POST %s parsed_json=%s", full_url, parsed)
        return parsed
