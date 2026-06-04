"""
Provider-swappable LLM client with automatic fallback chain.

Fallback order: Groq → Cerebras → Gemini.

Each provider supports:
  - Async generation via `generate()`
  - Optional JSON-schema constrained output
  - Sync convenience wrapper via `generate_sync()`

Usage::

    from backend.config import get_settings
    from backend.llm_client import create_fallback_client

    client = create_fallback_client(get_settings())
    response = await client.generate([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

from backend.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

MessageList = list[dict[str, str]]


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return True when *exc* looks like a 429 / rate-limit error."""
    # Check for an HTTP status attribute (groq, openai SDKs)
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    # Groq / OpenAI raise typed exceptions
    cls_name = type(exc).__name__.lower()
    if "ratelimit" in cls_name:
        return True
    # Fall back to inspecting the message
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Interface every LLM provider must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @abstractmethod
    async def generate(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Send *messages* to the model and return the assistant response text.

        Parameters
        ----------
        messages:
            OpenAI-style list of ``{"role": ..., "content": ...}`` dicts.
        json_schema:
            If provided, instruct the model to return output conforming to
            this JSON Schema.  Support is best-effort per provider.
        """

    # Sync convenience wrapper
    def generate_sync(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Blocking wrapper around :meth:`generate`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're inside an existing event loop (e.g. Jupyter / FastAPI).
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.generate(messages, json_schema=json_schema),
                ).result()
        return asyncio.run(self.generate(messages, json_schema=json_schema))


# ──────────────────────────────────────────────────────────────────────────────
# Groq provider
# ──────────────────────────────────────────────────────────────────────────────


class GroqProvider(LLMProvider):
    """Groq cloud inference using the official ``groq`` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "Groq"

    async def generate(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_object",
                "schema": json_schema,  # Groq supports schema-guided JSON
            }

        logger.debug("Groq request – model=%s, msgs=%d", self._model, len(messages))
        response = await self._client.chat.completions.create(**kwargs)
        text: str = response.choices[0].message.content or ""
        logger.debug("Groq response – %d chars", len(text))
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Cerebras provider (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────


class CerebrasProvider(LLMProvider):
    """Cerebras inference via the OpenAI-compatible API."""

    _BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(self, api_key: str, model: str) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=self._BASE_URL)
        self._model = model

    @property
    def name(self) -> str:
        return "Cerebras"

    async def generate(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": json_schema,
                },
            }

        logger.debug(
            "Cerebras request – model=%s, msgs=%d", self._model, len(messages)
        )
        response = await self._client.chat.completions.create(**kwargs)
        text: str = response.choices[0].message.content or ""
        logger.debug("Cerebras response – %d chars", len(text))
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Gemini provider
# ──────────────────────────────────────────────────────────────────────────────


class GeminiProvider(LLMProvider):
    """Google Gemini inference using the ``google-genai`` SDK."""

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return "Gemini"

    async def generate(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        from google.genai import types

        # google-genai expects a list of Content objects or plain dicts.
        contents: list[types.Content] = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                )
            )

        config_kwargs: dict[str, Any] = {}
        if json_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        logger.debug(
            "Gemini request – model=%s, msgs=%d", self._model, len(messages)
        )

        # google-genai's async API
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        text: str = response.text or ""
        logger.debug("Gemini response – %d chars", len(text))
        return text


# ──────────────────────────────────────────────────────────────────────────────
# Fallback client
# ──────────────────────────────────────────────────────────────────────────────


class FallbackLLMClient(LLMProvider):
    """Tries providers in order; falls through on rate-limit errors."""

    def __init__(self, providers: Sequence[LLMProvider], max_retries: int = 5) -> None:
        if not providers:
            raise ValueError("At least one LLM provider is required")
        self._providers: list[LLMProvider] = list(providers)
        self._max_retries = max_retries

    @property
    def name(self) -> str:
        names = ", ".join(p.name for p in self._providers)
        return f"Fallback[{names}]"

    async def generate(
        self,
        messages: MessageList,
        *,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        """Try each provider in order, robust to rate limits and dead providers.

        Within a provider we retry rate-limit (429) errors with exponential
        backoff — important when only one provider is actually usable. Any
        non-rate-limit error (e.g. a misconfigured model) skips to the next
        provider instead of crashing the whole request.
        """
        last_exc: Exception | None = None
        for provider in self._providers:
            delay = 2.0
            for attempt in range(self._max_retries):
                try:
                    return await provider.generate(messages, json_schema=json_schema)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if _is_rate_limit_error(exc):
                        logger.warning(
                            "Rate-limited by %s (attempt %d/%d); backing off %.1fs.",
                            provider.name, attempt + 1, self._max_retries, delay,
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * 2, 30.0)
                        continue
                    logger.warning(
                        "Provider %s failed (%s); trying next provider.",
                        provider.name, str(exc)[:120],
                    )
                    break  # non-rate-limit error -> next provider

        assert last_exc is not None
        raise RuntimeError("All LLM providers exhausted") from last_exc


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


def create_fallback_client(settings: Settings | None = None) -> FallbackLLMClient:
    """Build a :class:`FallbackLLMClient` from application settings.

    Only providers whose API keys are configured will be included in the
    fallback chain.  Order: Groq → Cerebras → Gemini.
    """
    if settings is None:
        settings = get_settings()

    providers: list[LLMProvider] = []

    if settings.groq_api_key:
        providers.append(GroqProvider(settings.groq_api_key, settings.groq_model))
        logger.info("Registered LLM provider: Groq (%s)", settings.groq_model)

    if settings.cerebras_api_key:
        providers.append(
            CerebrasProvider(settings.cerebras_api_key, settings.cerebras_model)
        )
        logger.info("Registered LLM provider: Cerebras (%s)", settings.cerebras_model)

    if settings.gemini_api_key:
        providers.append(GeminiProvider(settings.gemini_api_key, settings.gemini_model))
        logger.info("Registered LLM provider: Gemini (%s)", settings.gemini_model)

    if not providers:
        raise RuntimeError(
            "No LLM provider API keys configured. "
            "Set at least one of GROQ_API_KEY, CEREBRAS_API_KEY, or GEMINI_API_KEY "
            "in your .env file."
        )

    return FallbackLLMClient(providers)
