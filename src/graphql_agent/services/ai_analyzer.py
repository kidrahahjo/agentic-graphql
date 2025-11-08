from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import google.genai as genai
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from graphql_agent.config import get_settings


@dataclass
class ToolSuggestion:
    tool_name: str
    internal_final_outcome: str
    server_name: str
    confidence: float
    reasoning: str
    arguments: Any


class AIAnalyzer:
    def __init__(self) -> None:
        settings = get_settings()
        self._openai_client = (
            AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        )
        self._google_genai_client = (
            genai.Client(api_key=settings.google_genai_api_key)
            if settings.google_genai_api_key
            else None
        )
        self._openai_model = settings.openai_model
        self._google_genai_model = settings.google_genai_model

    async def analyze_prompt(
        self,
        base_prompt: str,
        available_servers_summary: list[str],
        existing_context: str,
    ) -> ToolSuggestion | None:
        prompt = base_prompt + ("\n" + existing_context if existing_context else "")
        return await self._analyze_with_google_genai(prompt, available_servers_summary)

    def _build_prompts(self, prompt: str, available_servers_summary: list[str]) -> tuple[str, str]:
        tools_summary = "\n".join(available_servers_summary)

        system_prompt = """You are an intelligent tool routing assistant.
Respond ONLY with valid JSON in this exact format:

{
  "server_name": "name of the server, empty if no server fits",
  "tool_name": "name of the tool, empty if no tool fits",
  "confidence": 0.0-1.0,
  "reasoning": "why this tool fits best",
  "suggested_arguments": {...},
  "internal_final_outcome": "optional final result if no further tool calls are needed"
}"""

        user_prompt = f"""User prompt: "{prompt}"

Summary of available servers and tools:
{tools_summary}"""

        return system_prompt, user_prompt

    def _parse_ai_response(self, content: str | None) -> ToolSuggestion | None:
        if not content:
            return None

        # Try to extract JSON even if extra text exists
        try:
            json_str = content.strip()
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start == -1 or end == -1:
                return None
            result = json.loads(json_str[start:end])
        except json.JSONDecodeError:
            return None

        return ToolSuggestion(
            tool_name=result.get("tool_name", ""),
            server_name=result.get("server_name", ""),
            confidence=float(result.get("confidence", 0.0)),
            reasoning=result.get("reasoning", ""),
            arguments=result.get("suggested_arguments", {}),
            internal_final_outcome=result.get("internal_final_outcome", ""),
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _analyze_with_openai(
        self,
        prompt: str,
        available_tools: list[str],
    ) -> ToolSuggestion | None:
        if self._openai_client is None:
            return None

        system_prompt, user_prompt = self._build_prompts(prompt, available_tools)

        response = await asyncio.wait_for(
            self._openai_client.chat.completions.create(
                model=self._openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            ),
            timeout=30,
        )
        content = response.choices[0].message.content
        return self._parse_ai_response(content)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _analyze_with_google_genai(
        self,
        prompt: str,
        available_servers_summary: list[str],
    ) -> ToolSuggestion | None:
        google_client = self._google_genai_client
        if google_client is None:
            return None

        system_prompt, user_prompt = self._build_prompts(prompt, available_servers_summary)

        config = genai.types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        )

        contents = [
            {"role": "model", "parts": [{"text": system_prompt}]},
            {"role": "user", "parts": [{"text": user_prompt}]},
        ]

        loop = asyncio.get_running_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: google_client.models.generate_content(
                    model=self._google_genai_model,
                    contents=contents,
                    config=config,
                ),
            ),
            timeout=45,
        )

        return self._parse_ai_response(response.text)
