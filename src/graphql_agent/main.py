from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Mount
from strawberry.asgi import GraphQL

from .config import get_settings
from .prompt_router import MCPServerBinding, PromptRouter, build_router
from .schema import schema
from .services.ai_analyzer import AIAnalyzer
from .services.mcp import MCPClient


class GlobalApplicationState:
    def __init__(self) -> None:
        self.settings = get_settings()

        self._mcp_clients: list[MCPClient] = []
        self.mcp_bindings: list[MCPServerBinding] = []
        for server in self.settings.mcp_servers:
            client = MCPClient(
                base_url=str(server.base_url),
                auth_token=self.settings.api_auth_token,
            )
            self._mcp_clients.append(client)
            binding = MCPServerBinding(
                name=server.name,
                client=client,
            )
            self.mcp_bindings.append(binding)

        if not self.mcp_bindings:
            raise RuntimeError("At least one MCP server must be configured")

        self.ai_analyzer = AIAnalyzer()

        self.router: PromptRouter = build_router(
            mcp_bindings=self.mcp_bindings,
            ai_analyzer=self.ai_analyzer,
        )

    async def build_context(self) -> dict:
        return {
            "router": self.router,
            "settings": self.settings,
        }

    async def close(self) -> None:
        for client in self._mcp_clients:
            await client.close()


state = GlobalApplicationState()


class StatefulGraphQL(GraphQL):
    def __init__(self, state: GlobalApplicationState) -> None:
        self._state = state
        super().__init__(schema)

    async def get_context(self, request, response) -> dict[str, Any]:
        context = await self._state.build_context()
        context.setdefault("request", request)
        context.setdefault("response", response)
        return context


graphql_app = StatefulGraphQL(state)


@asynccontextmanager
async def lifespan(_: Any) -> AsyncGenerator[Any, Any]:
    try:
        yield
    finally:
        await state.close()


app = Starlette(
    routes=[Mount("/graphql", graphql_app)],
    lifespan=lifespan,
)
