"""GraphQL schema definition."""

from __future__ import annotations

import json
import logging

import strawberry
from strawberry.scalars import JSON
from strawberry.types import Info

from .prompt_router import PromptRouter

LOGGER = logging.getLogger(__name__)


@strawberry.type
class AskPayload:
    """Result returned by the `ask` query."""

    content: str
    metadata: JSON


def get_router(info: Info) -> PromptRouter:
    router = info.context.get("router")
    if not isinstance(router, PromptRouter):  # pragma: no cover - runtime guard
        raise RuntimeError("Prompt router missing from GraphQL context")
    return router


@strawberry.type
class Query:
    @strawberry.field
    async def ask(self, info: Info, prompt: str) -> AskPayload:
        LOGGER.info("GraphQL ask query received: prompt_length=%d", len(prompt))
        LOGGER.debug("GraphQL ask query prompt: %s", prompt)
        router = get_router(info)
        LOGGER.debug("Router obtained from context")
        outcome = await router.dispatch(prompt)
        LOGGER.info(
            "Router returned outcome: intent=%s content_length=%d",
            outcome.metadata.get("intent"),
            len(outcome.content),
        )
        try:
            metadata = json.loads(json.dumps(outcome.metadata))
        except TypeError:
            LOGGER.warning("Failed to serialize metadata, using fallback")
            metadata = {"warning": "metadata not serializable"}
        LOGGER.debug("Returning AskPayload with metadata keys: %s", list(metadata.keys()))
        return AskPayload(content=outcome.content, metadata=metadata)


schema = strawberry.Schema(query=Query)
