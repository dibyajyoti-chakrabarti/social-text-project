from types import SimpleNamespace

import httpx2
import openai
import pytest

from frminer.llm import LLM, InvalidJSON, QuotaExceeded, parse_json
from frminer.llm.cache import Cache


class FakeClient:
    """Stands in for openai.OpenAI; returns queued replies and records calls."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def make_llm(replies, daily_limit=None):
    llm = LLM("primary", cache=Cache(":memory:"), client=FakeClient(replies))
    llm.daily_limit = daily_limit
    llm._throttle.interval = 0
    return llm


def rate_limit_error():
    request = httpx2.Request("POST", "https://example.com")
    response = httpx2.Response(429, request=request, headers={"retry-after": "0"})
    return openai.RateLimitError("slow down", response=response, body=None)


def test_second_identical_call_is_cached():
    llm = make_llm(["feature_request"])
    first = llm.complete("pls add a sleep timer")
    second = llm.complete("pls add a sleep timer")
    assert first.text == second.text == "feature_request"
    assert not first.cached and second.cached
    assert len(llm._client.calls) == 1


def test_different_system_prompt_is_a_different_cache_entry():
    llm = make_llm(["a", "b"])
    assert llm.complete("x", system="s1").text == "a"
    assert llm.complete("x", system="s2").text == "b"


def test_retries_on_rate_limit_then_succeeds():
    llm = make_llm([rate_limit_error(), "ok"])
    assert llm.complete("hi").text == "ok"
    assert llm.cache.requests_today(llm.provider, llm.quota_tz) == 2  # failed attempts count too


def test_daily_quota_enforced():
    llm = make_llm(["a", "b"], daily_limit=1)
    llm.complete("one")
    with pytest.raises(QuotaExceeded):
        llm.complete("two")


def test_complete_json_reasks_on_bad_json():
    llm = make_llm(["not json at all", '{"label": "bug_report"}'])
    data, resp = llm.complete_json("classify")
    assert data == {"label": "bug_report"}
    assert llm._client.calls[0]["response_format"] == {"type": "json_object"}


def test_complete_json_gives_up():
    llm = make_llm(["nope"] * 3)
    with pytest.raises(InvalidJSON):
        llm.complete_json("classify", json_attempts=3)


def test_map_preserves_order_and_isolates_errors():
    llm = make_llm(['{"i": 1}', "bad", "bad", "bad"])
    # Run sequentially so the queued replies map to prompts deterministically.
    results = llm.map(["p1", "p2"], json_mode=True, workers=1)
    assert results[0][0] == {"i": 1}
    assert isinstance(results[1], InvalidJSON)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("Sure! Here you go: [1, 2] hope that helps", [1, 2]),
    ],
)
def test_parse_json_tolerates_wrapping(text, expected):
    assert parse_json(text) == expected


def test_quota_counted_per_provider_timezone():
    cache = Cache(":memory:")
    cache.record_request("p", "Pacific/Kiritimati")  # UTC+14
    cache.record_request("p", "Pacific/Pago_Pago")  # UTC-11: always a different date
    assert cache.requests_today("p", "Pacific/Kiritimati") == 1
    assert cache.requests_today("p", "Pacific/Pago_Pago") == 1
