"""Application configuration and settings management."""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import BaseModel, Field, HttpUrl
from pydantic.functional_validators import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerSettings(BaseModel):
    """Configuration describing an MCP server and its schema."""

    name: str
    base_url: HttpUrl
    description: str | None = Field(
        default=None,
        description="Human-readable summary of the server's capabilities",
    )
    discover_path: str = Field(
        default="/mcp/discover",
        description="JSON-RPC 2.0 endpoint path for discovery",
    )
    invoke_path: str = Field(
        default="/mcp/invoke",
        description="JSON-RPC 2.0 endpoint path for prompt invocation",
    )


class Settings(BaseSettings):
    """Environment-driven configuration for the GraphQL agent."""

    api_auth_token: str | None = Field(
        default=None,
        description="Bearer token or API key added to downstream requests when present",
    )
    mcp_servers: list[MCPServerSettings] = Field(
        default_factory=list,
        description="List of MCP servers to consider for prompt routing",
        alias="MCP_SERVERS",
    )
    log_level: str = Field(
        default="INFO",
        description="Root logger level",
        alias="LOG_LEVEL",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for AI-powered prompt analysis (if provider=openai)",
        alias="OPENAI_API_KEY",
    )
    google_genai_api_key: str | None = Field(
        default=None,
        description="Google GenAI API key for AI-powered prompt analysis (if provider=google)",
        alias="GOOGLE_GENAI_API_KEY",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model to use for prompt analysis (if provider=openai)",
        alias="OPENAI_MODEL",
    )
    google_genai_model: str = Field(
        default="gemini-2.5-flash",
        description="Google GenAI model to use for prompt analysis (if provider=google)",
        alias="GOOGLE_GENAI_MODEL",
    )

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _parse_mcp_servers(cls, value: object) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:  # pragma: no cover - env misconfiguration
                raise ValueError("MCP_SERVERS must be valid JSON") from exc
        return value

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        case_sensitive=False,
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance.

    Using LRU caching keeps a single settings object per process.
    """

    return Settings()  # type: ignore[call-arg]
