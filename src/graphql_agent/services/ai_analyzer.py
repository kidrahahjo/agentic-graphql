"""AI-powered prompt analysis for tool selection and action planning."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import google.genai as genai
from openai import AsyncOpenAI

from graphql_agent.config import get_settings

LOGGER = logging.getLogger(__name__)


@dataclass
class ToolSuggestion:
    """AI-suggested tool and actions for a prompt."""

    tool_name: str
    internal_final_outcome: str
    server_name: str
    confidence: float
    reasoning: str
    arguments: Any


class AIAnalyzer:
    """Analyzes prompts using AI to suggest the best tool and actions."""

    def __init__(
        self,
    ) -> None:
        settings = get_settings()
        self._openai_client = None
        self._google_genai_client = None
        self._openai_model = settings.openai_model
        self._google_genai_model = settings.google_genai_model

        if settings.openai_api_key:
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

        if settings.google_genai_api_key:
            self._google_genai_client = genai.Client(api_key=get_settings().google_genai_api_key)

    async def analyze_prompt(
        self,
        base_prompt: str,
        available_tools: list[dict[str, Any]],
        existing_context: str,
    ) -> ToolSuggestion | None:
        """Analyze prompt and suggest the best tool and actions."""
        prompt = base_prompt
        if existing_context:
            prompt = prompt + "\n" + existing_context
        return await self._analyze_with_google_genai(prompt, available_tools)

    def _build_prompts(self, prompt: str, available_tools: list[dict[str, Any]]) -> tuple[str, str]:
        """Build system and user prompts for AI analysis."""
        tools_summary = [
            {
                "name": tool.get("name", "unknown"),
                "description": tool.get("description", ""),
                "tools": tool.get("tools", []),
            }
            for tool in available_tools
        ]

        system_prompt = """You are an intelligent tool routing assistant.
Your job is to:
1. Analyze the user's prompt.
2. Identify which server and tool are most appropriate.
3. Provide structured reasoning and arguments.

Respond ONLY with **valid JSON** in this exact format:

{
  "server_name": "name of the server, empty if no server fits",
  "tool_name": "name of the tool, empty if no tool fits",
  "confidence": 0.0-1.0,
  "reasoning": "why this tool fits best",
  "suggested_arguments": {...}, 
  "internal_final_outcome": "optional final result if no further tool calls are needed (answer as if you are giving the correct answer to the user)"
}"""

        user_prompt = f"""User prompt: "{prompt}"

Available servers and tools:
{json.dumps(tools_summary, indent=2)}
Choose the most relevant tool and fill the JSON fields accordingly."""

        return system_prompt, user_prompt

    def _parse_ai_response(self, content: Any) -> ToolSuggestion | None:
        """Parse AI response and create ToolSuggestion."""
        if not content:
            return None

        try:
            result = json.loads(content)
            suggestion = ToolSuggestion(
                tool_name=result.get("tool_name", ""),
                server_name=result.get("server_name", ""),
                confidence=float(result.get("confidence", 0.0)),
                reasoning=result.get("reasoning", ""),
                arguments=result.get("suggested_arguments", []),
                internal_final_outcome=result.get("internal_final_outcome", ""),
            )
            LOGGER.info(
                "AI suggested tool: %s (confidence=%.2f, arguments=%d)",
                suggestion.tool_name,
                suggestion.confidence,
                len(suggestion.arguments),
            )
            return suggestion
        except json.JSONDecodeError:
            LOGGER.error("Failed to parse AI response as JSON: %s", content[:200])
            return None

    async def _analyze_with_openai(
        self,
        prompt: str,
        available_tools: list[dict[str, Any]],
    ) -> ToolSuggestion | None:
        if self._openai_client is None:
            LOGGER.exception("OpenAI client not initialized")
            return None

        LOGGER.info(
            "Analyzing prompt with OpenAI (model=%s, tools=%d)",
            self._openai_model,
            len(available_tools),
        )
        system_prompt, user_prompt = self._build_prompts(prompt, available_tools)

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            LOGGER.debug("OpenAI analysis response: %s", content)
            return self._parse_ai_response(content)
        except Exception as exc:
            LOGGER.exception("OpenAI analysis failed: %s", exc)
            return None

    async def _analyze_with_google_genai(
        self,
        prompt: str,
        available_tools: list[dict[str, Any]],
    ) -> ToolSuggestion | None:
        if self._google_genai_client is None:
            LOGGER.exception("Google GenAI Client not initialized")
            return None

        LOGGER.info(
            "Analyzing prompt with Google GenAI (model=%s, tools=%d)",
            self._google_genai_model,
            len(available_tools),
        )
        system_prompt, user_prompt = self._build_prompts(prompt, available_tools)

        try:
            config = genai.types.GenerateContentConfig(
                temperature=0.3, response_mime_type="application/json"
            )

            contents = [
                {"role": "model", "parts": [{"text": system_prompt}]},
                {"role": "user", "parts": [{"text": user_prompt}]},
            ]

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._google_genai_client.models.generate_content(
                    model=self._google_genai_model, contents=contents, config=config
                ),
            )
            content = response.text
            LOGGER.debug("Google GenAI analysis response: %s", content)
            return self._parse_ai_response(content)
        except Exception as exc:
            LOGGER.exception("Google GenAI analysis failed: %s", exc)
            return None
