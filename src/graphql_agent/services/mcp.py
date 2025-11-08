from __future__ import annotations

from typing import Any, overload

import httpx

from graphql_agent.types.mcp_2025_08_16 import (
    CallToolRequest,
    CallToolResponse,
    InitializeRequest,
    InitializeResponse,
    ListToolsRequest,
    ListToolsResponse,
    make_call_tool_request,
    make_initialize_request,
    make_list_tools_request,
)

from .base import JSONServiceClient


class MCPClient(JSONServiceClient):
    @overload
    async def _rpc_call(
        self, request_object: InitializeRequest, headers: dict[str, Any]
    ) -> InitializeResponse: ...
    @overload
    async def _rpc_call(
        self, request_object: ListToolsRequest, headers: dict[str, Any]
    ) -> ListToolsResponse: ...
    @overload
    async def _rpc_call(
        self, request_object: CallToolRequest, headers: dict[str, Any]
    ) -> CallToolResponse: ...

    async def _rpc_call(
        self,
        request_object: InitializeRequest | ListToolsRequest | CallToolRequest,
        headers: dict[str, Any],
    ) -> InitializeResponse | ListToolsResponse | CallToolResponse:
        response = await self._client.post(
            "/mcp",
            json=request_object,
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers=headers,
        )
        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON from MCP server: {e}") from e

        if payload.get("error"):
            raise RuntimeError(f"MCP error: {payload['error']}")

        return payload

    async def initialize(self) -> InitializeResponse:
        initialize_request = make_initialize_request()
        try:
            initialize_response = await self._rpc_call(
                request_object=initialize_request,
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError("Failed to query MCP server") from exc

        response_version = initialize_response.get("result", {}).get("protocolVersion")
        request_version = initialize_request.get("params", {}).get("protocolVersion")
        if response_version != request_version:
            raise RuntimeError(
                f"Protocol version mismatch. Supported: {request_version}. Got: {response_version}"
            )

        return initialize_response

    async def list_tools(self) -> ListToolsResponse:
        request = make_list_tools_request()
        try:
            response = await self._rpc_call(
                request_object=request,
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError("Failed to fetch tools from the MCP server") from exc

        return response

    async def call_tool(
        self,
        user_token: str,
        tool_name: str | None = None,
        arguments: Any = None,
    ) -> CallToolResponse:
        request = make_call_tool_request(
            name=tool_name,
            arguments=arguments,
        )

        try:
            return await self._rpc_call(
                request_object=request,
                headers={
                    "authorization": user_token,
                    "x-authorization": user_token,
                },
            )

        except httpx.HTTPStatusError:
            raise
