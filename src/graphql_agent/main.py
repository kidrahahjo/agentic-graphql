"""ASGI entrypoint for the GraphQL agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from starlette.applications import Starlette
from starlette.routing import Mount
from strawberry.asgi import GraphQL

from .config import get_settings
from .context import UserContext, UserContextLoader
from .prompt_router import MCPServerBinding, PromptRouter, build_router
from .schema import schema
from .services.ai_analyzer import AIAnalyzer
from .services.mcp import MCPClient


class ApplicationState:
    """State container that wires settings, clients, and router."""

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
                description=server.description,
            )
            binding.discover_path = server.discover_path  # type: ignore[attr-defined]
            binding.invoke_path = server.invoke_path  # type: ignore[attr-defined]
            self.mcp_bindings.append(binding)

        if not self.mcp_bindings:
            raise RuntimeError("At least one MCP server must be configured")

        self.user_context_loader = UserContextLoader()

        self.ai_analyzer = AIAnalyzer()

        self.router: PromptRouter = build_router(
            mcp_bindings=self.mcp_bindings,
            user_loader=self.user_context_loader,
            ai_analyzer=self.ai_analyzer,
        )

    async def build_context(self) -> UserContext:
        context = await self.user_context_loader.load()
        context.update(
            {
                "router": self.router,
                "settings": self.settings,
            }
        )
        return context

    async def close(self) -> None:
        for client in self._mcp_clients:
            await client.close()


state = ApplicationState()


class StatefulGraphQL(GraphQL):
    """GraphQL ASGI app wired with our shared application state."""

    def __init__(self, state: ApplicationState) -> None:
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
