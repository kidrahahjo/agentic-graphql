"""Prompt routing logic determining how to satisfy a query."""

from __future__ import annotations

import abc
import asyncio
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .context import UserContextLoader
from .services.ai_analyzer import AIAnalyzer
from .services.mcp import MCPClient

logger = logging.getLogger(__name__)


PROMPT_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(slots=True)
class PromptOutcome:
    """Value returned by prompt strategies."""

    content: str
    metadata: Mapping[str, Any]


class PromptStrategy(abc.ABC):
    """Strategy interface for prompt routing."""

    @abc.abstractmethod
    def matches(self, prompt: str) -> bool:
        """Return True if the strategy should handle the prompt."""

    @abc.abstractmethod
    async def execute(self, prompt: str) -> PromptOutcome:
        """Process the prompt and return a response."""


def _tokenise(value: str) -> list[str]:
    return [token.lower() for token in PROMPT_TOKEN_RE.findall(value.lower())]


@dataclass(slots=True)
class MCPServerBinding:
    """Binding between a configured MCP server and its derived metadata."""

    name: str
    client: MCPClient
    description: str | None
    discover_path: str = "/mcp/discover"
    invoke_path: str = "/mcp/invoke"
    _mcp_tools: list[dict] = field(default_factory=list, init=False)
    _schema_loaded: bool = field(default=False, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    async def resolved_mcp_tools(self) -> list[dict]:
        if self._schema_loaded:
            logger.debug(
                "Server %s: using cached keywords (schema=%d)", self.name, len(self._mcp_tools)
            )
            return self._mcp_tools

        logger.info("Server %s: loading schema", self.name)
        async with self._lock:
            if not self._schema_loaded:
                self._mcp_tools = await self._load_mcp_tools()
                self._schema_loaded = True
                logger.info("Server %s: loaded %d schema keywords", self.name, len(self._mcp_tools))

        return self._mcp_tools

    async def _load_mcp_tools(self) -> Any:
        try:
            logger.info("Server %s: discovering schema via %s", self.name, self.discover_path)
            # Discover via JSON-RPC to obtain schema/capabilities
            discover_result = await self.client.discover(self.discover_path)
            # Accept either a top-level "schema" or the whole result if schema is root
            tools = discover_result.get("tools", [])
            return tools
        except Exception:  # pragma: no cover - log and continue
            logger.warning("Failed to obtain schema for MCP server '%s'", self.name, exc_info=True)
            return []


class SchemaAwareMCPStrategy(PromptStrategy):
    """Delegates prompts to the most relevant MCP server."""

    def __init__(
        self,
        servers: Sequence[MCPServerBinding],
        context_loader: UserContextLoader,
        ai_analyzer: AIAnalyzer,
    ) -> None:
        if not servers:
            raise ValueError("At least one MCP server must be configured")

        self._servers = list(servers)
        self._context_loader = context_loader
        self._ai_analyzer = ai_analyzer

    def matches(self, prompt: str) -> bool:
        return True

    async def execute(self, prompt: str) -> PromptOutcome:
        logger.info("Executing prompt routing for: %s", prompt)
        prompt_tokens = _tokenise(prompt)
        logger.debug("Tokenized prompt into %d tokens: %s", len(prompt_tokens), prompt_tokens)
        if not prompt_tokens:
            raise ValueError("Prompt must contain at least one token")

        available_servers = [
            {
                "name": binding.name,
                "description": binding.description or "",
                "tools": list(await binding.resolved_mcp_tools()),
            }
            for binding in self._servers
        ]

        # Its allowed to do at max 10 API calls
        max_attempts = 10
        existing_context = ""

        while max_attempts:
            max_attempts -= 1

            ai_suggestion = await self._ai_analyzer.analyze_prompt(
                prompt, available_servers, existing_context
            )
            print(ai_suggestion)
            if ai_suggestion and ai_suggestion.internal_final_outcome:
                return PromptOutcome(
                    content=ai_suggestion.internal_final_outcome,
                    metadata={
                        "intent": "finished",
                        "confidence": ai_suggestion.confidence,
                        "num_tries": 10 - max_attempts,
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

            logger.info(
                "AI suggested tool: %s (confidence=%.2f, reasoning=%s)",
                ai_suggestion.tool_name,
                ai_suggestion.confidence,
                ai_suggestion.reasoning,
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
                result = await binding.client.handle(
                    prompt=prompt,
                    tool_name=ai_suggestion.tool_name,
                    arguments=ai_suggestion.arguments,
                )
                existing_context = (
                    existing_context
                    + "\n"
                    + f"Response from server {ai_suggestion.server_name} and tool {ai_suggestion.tool_name}: {result}"
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
    """Coordinate prompt handling using a set of strategies."""

    def __init__(self, strategies: Iterable[PromptStrategy]) -> None:
        self._strategies = list(strategies)

    async def dispatch(self, prompt: str) -> PromptOutcome:
        logger.info("Router.dispatch called with prompt length=%d", len(prompt))
        normalized = prompt.strip()
        if not normalized:
            raise ValueError("Prompt must not be empty")

        logger.info("Evaluating %d strategy/strategies", len(self._strategies))
        for idx, strategy in enumerate(self._strategies):
            logger.debug("Trying strategy %d: %s", idx, strategy.__class__.__name__)
            try:
                if strategy.matches(normalized):
                    logger.info("Strategy %s matched, executing...", strategy.__class__.__name__)
                    outcome = await strategy.execute(normalized)
                    logger.info("Strategy %s completed successfully", strategy.__class__.__name__)
                    return outcome
                else:
                    logger.debug("Strategy %s did not match", strategy.__class__.__name__)
            except Exception:  # pragma: no cover - surface errors with context
                logger.exception("Strategy %s failed", strategy.__class__.__name__)
                raise

        logger.error("No strategy handled the prompt")
        raise RuntimeError("No strategy handled the prompt")


def build_router(
    *,
    mcp_bindings: Sequence[MCPServerBinding],
    user_loader: UserContextLoader,
    ai_analyzer: AIAnalyzer,
) -> PromptRouter:
    """Construct router with default strategies."""

    strategies: list[PromptStrategy] = [
        SchemaAwareMCPStrategy(
            servers=mcp_bindings, context_loader=user_loader, ai_analyzer=ai_analyzer
        )
    ]

    return PromptRouter(strategies=strategies)
