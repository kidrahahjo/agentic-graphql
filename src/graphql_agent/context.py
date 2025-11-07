"""Helper utilities for populating request context."""

from __future__ import annotations

from typing import TypedDict


class UserContext(TypedDict, total=False):
    id: str
    email: str
    username: str
    display_name: str


class UserContextLoader:
    """Provide user/request context for downstream calls (no external deps)."""

    def __init__(self) -> None:
        pass

    async def load(self) -> UserContext:
        # Intentionally empty: context can be enriched later from request headers
        # or by a custom middleware without introducing hard-coded external calls.
        return {}
