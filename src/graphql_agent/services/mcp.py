"""Thin client for a Model Context Protocol server (JSON-RPC 2.0)."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from typing import Any

import httpx

from .base import JSONServiceClient

LOGGER = logging.getLogger(__name__)


class MCPClient(JSONServiceClient):
    """Delegate prompts to an MCP server via JSON-RPC 2.0."""

    _id_counter = itertools.count(1)

    async def _rpc_call(
        self, path: str, method: str, params: Mapping[str, Any] | None = None
    ) -> Any:
        request_id = next(self._id_counter)
        LOGGER.debug("RPC request %s %s id=%s", method, path, request_id)
        request = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        response = await self._client.post(path, json=request, headers=self._headers())
        response.raise_for_status()
        payload = response.json()
        LOGGER.debug("RPC response %s id=%s status=%s", method, request_id, response.status_code)
        if "error" in payload and payload["error"] is not None:
            LOGGER.error("MCP error for %s id=%s: %s", method, request_id, payload["error"])
            raise RuntimeError(f"MCP error: {payload['error']}")
        return payload

    async def discover(self, discover_path: str = "/mcp/discover") -> Any:
        """Call the discovery method to obtain schema/capabilities."""

        LOGGER.debug("Discovering MCP schema via %s", discover_path)
        try:
            return await self._rpc_call(discover_path, method="discover", params={})
        except httpx.HTTPStatusError as exc:
            LOGGER.exception(
                "Discover RPC failed with status %s;",
                exc.response.status_code,
            )
            return {}

    async def handle(
        self,
        prompt: str,
        *,
        context: Mapping[str, Any] | None = None,
        invoke_path: str = "/mcp/invoke",
        tool_name: str | None = None,
        arguments: Any = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if tool_name:
            params["name"] = tool_name
        if context:
            params["context"] = {
                "prompt": prompt,
            }
        if arguments:
            params["arguments"] = arguments

        LOGGER.debug(
            "Invoking MCP at %s with tool_name=%s context_keys=%s arguments=%d",
            invoke_path,
            tool_name,
            sorted(context.keys()) if context else [],
            arguments,
        )
        try:
            return await self._rpc_call(invoke_path, method="invoke", params=params)
        except httpx.HTTPStatusError as exc:
            LOGGER.warning(
                "Invoke RPC failed with status %s; attempting fallback POST",
                exc.response.status_code,
            )
            if exc.response.status_code in {404, 415, 422}:
                fallback_response = await self._client.post(
                    invoke_path,
                    json=params,
                    headers=self._headers(),
                )
                fallback_response.raise_for_status()
                LOGGER.debug("Fallback invoke succeeded status=%s", fallback_response.status_code)
                try:
                    return fallback_response.json()
                except ValueError as json_exc:  # pragma: no cover
                    LOGGER.exception("Fallback invoke response not valid JSON")
                    raise RuntimeError("MCP invoke response was not valid JSON") from json_exc
            raise
