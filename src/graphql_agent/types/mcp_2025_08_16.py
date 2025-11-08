from __future__ import annotations

from typing import Any, Literal, TypedDict

from graphql_agent import __version__
from graphql_agent.config import get_settings

# ==== Core Constants ====

JSONRPC_VERSION: str = "2.0"
LATEST_PROTOCOL_VERSION: str = "2025-06-18"

RequestId = str | int


# ==== Base Types ====


class Request(TypedDict, total=False):
    method: str
    params: dict[str, Any] | None


class Result(TypedDict, total=False):
    _meta: dict[str, Any] | None


class JSONRPCRequest(Request, total=False):
    jsonrpc: Literal["2.0"]
    id: RequestId


class JSONRPCResponse(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: RequestId
    result: Result


# ==== Shared Metadata ====


class BaseMetadata(TypedDict, total=False):
    name: str
    title: str | None


class Implementation(BaseMetadata, total=False):
    version: str


# ==== Capabilities ====


class ClientCapabilities(TypedDict, total=False):
    experimental: dict[str, object] | None
    sampling: dict[str, Any] | None
    elicitation: dict[str, Any] | None


class ServerCapabilities(TypedDict, total=False):
    experimental: dict[str, object] | None
    tools: dict[str, Any] | None


# ==== Initialize ====


class InitializeParams(TypedDict, total=False):
    capabilities: ClientCapabilities
    clientInfo: Implementation
    protocolVersion: str


class InitializeRequest(JSONRPCRequest, total=False):
    method: Literal["initialize"]
    params: InitializeParams


class InitializeResult(Result, total=False):
    protocolVersion: str
    capabilities: ServerCapabilities
    serverInfo: Implementation
    instructions: str | None


class InitializeResponse(JSONRPCResponse, total=False):
    result: InitializeResult


# ==== Tools ====


class ToolAnnotations(TypedDict, total=False):
    title: str | None
    readOnlyHint: bool | None
    destructiveHint: bool | None
    idempotentHint: bool | None
    openWorldHint: bool | None


class Tool(BaseMetadata, total=False):
    description: str | None
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None
    annotations: ToolAnnotations | None
    _meta: dict[str, Any] | None


# ==== tools/list ====


class ListToolsParams(TypedDict, total=False):
    cursor: str | None


class ListToolsRequest(JSONRPCRequest, total=False):
    method: Literal["tools/list"]
    params: ListToolsParams | None


class ListToolsResult(Result, total=False):
    tools: list[Tool]
    nextCursor: str | None


class ListToolsResponse(JSONRPCResponse, total=False):
    result: ListToolsResult


# ==== tools/call ====


class CallToolParams(TypedDict, total=False):
    name: str
    arguments: dict[str, Any] | None


class CallToolRequest(JSONRPCRequest, total=False):
    method: Literal["tools/call"]
    params: CallToolParams


class CallToolResult(Result, total=False):
    content: list[Any]
    structuredContent: dict[str, Any] | None
    isError: bool | None


class CallToolResponse(JSONRPCResponse, total=False):
    result: CallToolResult


# ==== Convenience Type Aliases ====

ClientRequest = InitializeRequest | ListToolsRequest | CallToolRequest

ServerResponse = InitializeResponse | ListToolsResponse | CallToolResponse


def make_initialize_request(
    capabilities: ClientCapabilities | None = None,
    protocol_version: str = LATEST_PROTOCOL_VERSION,
) -> InitializeRequest:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "initialize",
        "params": {
            "clientInfo": {"name": get_settings().application_name, "version": __version__},
            "capabilities": capabilities or {},
            "protocolVersion": protocol_version,
        },
    }


def make_list_tools_request(
    cursor: str | None = None,
) -> ListToolsRequest:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "tools/list",
        "params": {"cursor": cursor} if cursor is not None else None,
    }


def make_call_tool_request(
    name: str,
    arguments: dict[str, Any] | None = None,
) -> CallToolRequest:
    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": name,
            "arguments": arguments or {},
        },
    }
