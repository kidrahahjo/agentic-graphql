from __future__ import annotations

import abc
import asyncio
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from graphql_agent.types.mcp_2025_08_16 import InitializeResponse, Tool

from .services.ai_analyzer import AIAnalyzer
from .services.mcp import MCPClient

PROMPT_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokenise(value: str) -> list[str]:
    return [token.lower() for token in PROMPT_TOKEN_RE.findall(value.lower())]


@dataclass(slots=True)
class PromptOutcome:
    content: str
    metadata: Mapping[str, Any]


class PromptStrategy(abc.ABC):
    @abc.abstractmethod
    def matches(self, prompt: str) -> bool: ...

    @abc.abstractmethod
    async def execute(self, prompt: str, user_token: str) -> PromptOutcome: ...


@dataclass(slots=True)
class MCPServerBinding:
    name: str
    client: MCPClient
    _mcp_tools: list[Tool] = field(default_factory=list, init=False)
    _server_initialization: InitializeResponse = field(
        default_factory=InitializeResponse, init=False
    )
    _schema_loaded: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def build_context(self) -> str:
        await self.load_mcp_meta()
        return f"""[Server Name: {self.name}]
Information about the server: {self._server_initialization.get("result", {}).get("instructions", "") or "No extra information available for the server"}
Tools available for the server: {self._mcp_tools}
"""

    async def load_mcp_meta(self) -> None:
        if self._schema_loaded:
            return None

        async with self._lock:
            if not self._schema_loaded:
                await self._load_mcp_meta()
                self._schema_loaded = True

        return None

    async def _load_mcp_meta(self) -> None:
        try:
            self._server_initialization = await self.client.initialize()
        except Exception:
            self._server_initialization = InitializeResponse()

        try:
            # TODO: Pagination for list_tools is not supported as of now.
            self._mcp_tools = (await self.client.list_tools()).get("result", {}).get("tools", [])
        except Exception:
            self._mcp_tools = []


class SchemaAwareMCPStrategy(PromptStrategy):
    def __init__(
        self,
        servers: Sequence[MCPServerBinding],
        ai_analyzer: AIAnalyzer,
    ) -> None:
        if not servers:
            raise ValueError("At least one MCP server must be configured")

        self._servers = list(servers)
        self._ai_analyzer = ai_analyzer

    def matches(self, prompt: str) -> bool:
        return True

    async def execute(self, prompt: str, user_token: str) -> PromptOutcome:
        available_servers_summary = [await binding.build_context() for binding in self._servers]

        attempt = 0
        existing_context = ""

        # Its allowed to do at max 10 API calls
        while attempt < 10:
            attempt += 1

            ai_suggestion = await self._ai_analyzer.analyze_prompt(
                prompt,
                available_servers_summary,
                existing_context,
            )

            if ai_suggestion and ai_suggestion.internal_final_outcome:
                return PromptOutcome(
                    content=ai_suggestion.internal_final_outcome,
                    metadata={
                        "intent": "finished",
                        "confidence": ai_suggestion.confidence,
                        "num_tries": attempt,
                    },
                )

            if not ai_suggestion or ai_suggestion.confidence < 0.5:
                intent = "no_suggestion" if not ai_suggestion else "low_confidence"
                confidence = 0 if not ai_suggestion else ai_suggestion.confidence

                return PromptOutcome(
                    content="Sorry, I couldn't find what you're looking for.",
                    metadata={
                        "intent": intent,
                        "confidence": confidence,
                    },
                )

            binding: MCPServerBinding | None = None

            for server in self._servers:
                if server.name.lower() == ai_suggestion.server_name.lower():
                    binding = server
                    break

            if not binding:
                return PromptOutcome(
                    content="Sorry, I couldn't find what you're looking for.",
                    metadata={
                        "intent": "no_mcp_server_found",
                        "confidence": 0,
                    },
                )

            try:
                result = await binding.client.call_tool(
                    tool_name=ai_suggestion.tool_name,
                    arguments=ai_suggestion.arguments,
                    user_token=user_token,
                )
                existing_context = (
                    existing_context
                    + "\n"
                    + f"Response from server {ai_suggestion.server_name} and tool {ai_suggestion.tool_name}: {result.get('result', {})}"
                )
            except Exception as ex:
                return PromptOutcome(
                    content="Something went wrong while querying the MCP server",
                    metadata={
                        "intent": "mcp_server_failed",
                        "tool_name": ai_suggestion.tool_name,
                        "arguments": ai_suggestion.arguments,
                        "details": str(ex),
                        "confidence": 0,
                    },
                )

        return PromptOutcome(
            content="Could not find the results after a lot of attempts, please narrow down the query",
            metadata={
                "intent": "max_attempt_exceeded",
                "context_gathered": existing_context,
            },
        )


class PromptRouter:
    def __init__(self, strategies: Iterable[PromptStrategy]) -> None:
        self._strategies = list(strategies)

    async def dispatch(self, prompt: str, user_token: str) -> PromptOutcome:
        normalized = prompt.strip()
        prompt_tokens = _tokenise(normalized)

        if not normalized:
            raise ValueError("Prompt must not be empty")

        if not prompt_tokens:
            raise ValueError("Prompt must contain at least one token")

        for strategy in self._strategies:
            if strategy.matches(normalized):
                return await strategy.execute(normalized, user_token)

        raise RuntimeError("No strategy handled the prompt")


def build_router(
    *,
    mcp_bindings: Sequence[MCPServerBinding],
    ai_analyzer: AIAnalyzer,
) -> PromptRouter:
    strategies: list[PromptStrategy] = [
        SchemaAwareMCPStrategy(servers=mcp_bindings, ai_analyzer=ai_analyzer)
    ]

    return PromptRouter(strategies=strategies)
