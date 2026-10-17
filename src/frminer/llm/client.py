"""Multi-provider LLM client (Gemini, Groq, OpenRouter) with caching, throttling and quotas.

Usage:
    llm = LLM("primary")
    resp = llm.complete("Classify: 'pls add a sleep timer'")
    data, resp = llm.complete_json(prompt, system=SYSTEM)
    results = llm.map(prompts, system=SYSTEM, json_mode=True)

All providers are called through their OpenAI-compatible endpoints. Model choice
is by named profile (see `models:` in configs/config.yaml).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import openai
from tqdm import tqdm

from frminer.config import load_config, path
from frminer.llm.cache import Cache, request_key

RETRYABLE = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

Messages = list[dict[str, str]]


class QuotaExceeded(RuntimeError):
    """Raised when a provider's configured daily request limit is reached."""


class InvalidJSON(ValueError):
    """Raised when the model fails to return parseable JSON after all attempts."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    cached: bool


class _Throttle:
    """Spaces out request starts so a provider never exceeds `rpm` requests/minute."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next)
            self._next = start + self.interval
        time.sleep(max(0.0, start - now))


_throttles: dict[str, _Throttle] = {}
_throttles_lock = threading.Lock()
_default_cache: Cache | None = None


def _throttle_for(provider: str, rpm: float) -> _Throttle:
    with _throttles_lock:
        if provider not in _throttles:
            _throttles[provider] = _Throttle(rpm)
        return _throttles[provider]


def default_cache() -> Cache:
    global _default_cache
    if _default_cache is None:
        _default_cache = Cache(path("cache") / "llm.sqlite")
    return _default_cache


def parse_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating markdown fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the outermost {...} or [...] span.
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise InvalidJSON(f"Could not parse JSON from model output: {text[:200]!r}")


class LLM:
    def __init__(
        self,
        profile: str = "primary",
        cache: Cache | None = None,
        client: Any | None = None,
    ):
        cfg = load_config()
        prof = cfg["models"][profile]
        self.profile = profile
        self.provider = prof["provider"]
        self.model = prof["model"]
        self.params: dict[str, Any] = prof.get("params", {})
        pcfg = cfg["providers"][self.provider]
        self.daily_limit: int | None = pcfg.get("daily_limit")
        self.quota_tz: str = pcfg.get("quota_tz", "UTC")
        self.temperature: float = cfg["llm"]["temperature"]
        self.max_retries: int = cfg["llm"]["max_retries"]
        self.cache = cache or default_cache()
        self._throttle = _throttle_for(self.provider, pcfg["rpm"])
        if client is None:
            api_key = os.environ.get(pcfg["api_key_env"])
            if not api_key:
                raise RuntimeError(f"Missing env var {pcfg['api_key_env']} for {self.provider}")
            client = openai.OpenAI(
                base_url=pcfg["base_url"],
                api_key=api_key,
                timeout=cfg["llm"]["timeout_s"],
                max_retries=0,  # we handle retries ourselves to count quota correctly
            )
        self._client = client

    def __repr__(self) -> str:
        return f"LLM(profile={self.profile!r}, provider={self.provider!r}, model={self.model!r})"

    # ------------------------------------------------------------------ core

    def complete(
        self,
        prompt: str | Messages,
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        attempt: int = 0,
    ) -> LLMResponse:
        """One chat completion. `attempt` > 0 deliberately bypasses an earlier cached reply."""
        messages = self._messages(prompt, system)
        request = {
            "provider": self.provider,
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "json_mode": json_mode,
            "max_tokens": max_tokens,
            "params": self.params,
            "attempt": attempt,
        }
        key = request_key(request)
        hit = self.cache.get(key)
        if hit is not None:
            return LLMResponse(provider=self.provider, model=self.model, cached=True, **hit)

        text, usage, latency = self._call_with_retries(messages, json_mode, max_tokens)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        self.cache.put(key, request, text, prompt_tokens, completion_tokens, latency)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            cached=False,
        )

    def complete_json(
        self,
        prompt: str | Messages,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        json_attempts: int = 3,
        validate: Callable[[Any], None] | None = None,
    ) -> tuple[Any, LLMResponse]:
        """Completion parsed as JSON. Re-asks (uncached) on unparseable or invalid output.

        `validate` may raise ValueError to reject a structurally wrong reply.
        """
        last_err: Exception | None = None
        for attempt in range(json_attempts):
            resp = self.complete(
                prompt, system=system, json_mode=True, max_tokens=max_tokens, attempt=attempt
            )
            try:
                data = parse_json(resp.text)
                if validate is not None:
                    validate(data)
                return data, resp
            except ValueError as e:
                last_err = e
        raise InvalidJSON(f"{self!r} gave invalid JSON {json_attempts} times: {last_err}")

    def map(
        self,
        prompts: Sequence[str | Messages],
        *,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        validate: Callable[[Any], None] | None = None,
        workers: int = 4,
        desc: str | None = None,
    ) -> list[Any]:
        """Run many prompts concurrently (throttled). Returns results in input order.

        With json_mode, each result is `(data, LLMResponse)` or the exception raised for
        that prompt; without it, each result is an `LLMResponse` or an exception. Errors
        are returned rather than raised so one bad item never loses a whole batch.
        """

        def run(p: str | Messages) -> Any:
            try:
                if json_mode:
                    return self.complete_json(
                        p, system=system, max_tokens=max_tokens, validate=validate
                    )
                return self.complete(p, system=system, max_tokens=max_tokens)
            except (InvalidJSON, QuotaExceeded, openai.APIError) as e:
                return e

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(tqdm(pool.map(run, prompts), total=len(prompts), desc=desc or self.profile))

    # ------------------------------------------------------------- internals

    @staticmethod
    def _messages(prompt: str | Messages, system: str | None) -> Messages:
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        if system:
            messages = [{"role": "system", "content": system}, *messages]
        return messages

    def _check_quota(self) -> None:
        if self.daily_limit is not None:
            used = self.cache.requests_today(self.provider, self.quota_tz)
            if used >= self.daily_limit:
                raise QuotaExceeded(
                    f"{self.provider}: daily limit of {self.daily_limit} requests reached"
                )

    def _call_with_retries(
        self, messages: Messages, json_mode: bool, max_tokens: int | None
    ) -> tuple[str, Any, float]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.params,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(self.max_retries + 1):
            self._check_quota()
            self._throttle.wait()
            self.cache.record_request(self.provider, self.quota_tz)
            start = time.monotonic()
            try:
                resp = self._client.chat.completions.create(**kwargs)
            except RETRYABLE as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(self._backoff(attempt, e))
                continue
            latency = time.monotonic() - start
            text = resp.choices[0].message.content or ""
            return text, resp.usage, latency
        raise AssertionError("unreachable")

    @staticmethod
    def _backoff(attempt: int, err: Exception) -> float:
        response = getattr(err, "response", None)
        retry_after = response.headers.get("retry-after") if response is not None else None
        if retry_after:
            try:
                return min(float(retry_after), 120.0)
            except ValueError:
                pass
        return min(2**attempt * 2.0, 60.0) + random.uniform(0, 1)
