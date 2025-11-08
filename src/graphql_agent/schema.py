from __future__ import annotations

import json

import strawberry
from starlette.requests import Request
from strawberry.scalars import JSON
from strawberry.types import Info

from .prompt_router import PromptRouter


@strawberry.type
class AskPayload:
    content: str
    metadata: JSON


def get_router(info: Info) -> PromptRouter:
    router = info.context.get("router")
    if not isinstance(router, PromptRouter):
        raise RuntimeError("Prompt router missing from GraphQL context")
    return router


def get_user_token(info: Info) -> str:
    request: Request = info.context.get("request")
    valid_keys = ["authorization", "x-authorization"]
    provided_tokens = [
        request.headers.get(key, "") for key in valid_keys if request.headers.get(key, "")
    ]
    if provided_tokens:
        return provided_tokens[0]

    raise RuntimeError("User token must be supplied")


@strawberry.type
class Query:
    @strawberry.field
    async def ask(self, info: Info, prompt: str) -> AskPayload:
        router = get_router(info)
        user_token = get_user_token(info)
        outcome = await router.dispatch(prompt, user_token)
        try:
            metadata = json.loads(json.dumps(outcome.metadata))
        except TypeError:
            metadata = {"warning": "metadata not serializable"}
        return AskPayload(content=outcome.content, metadata=metadata)


schema = strawberry.Schema(query=Query)
