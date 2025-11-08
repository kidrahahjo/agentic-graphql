from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx


class ServiceClient:
    def __init__(self, base_url: str, auth_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=10.0)

    async def close(self) -> None:
        await self._client.aclose()


class JSONServiceClient(ServiceClient):
    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()

    async def post(
        self,
        path: str,
        payload: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        response = await self._client.post(
            path,
            json=payload,
            params=params,
        )
        response.raise_for_status()
        return response.json()
