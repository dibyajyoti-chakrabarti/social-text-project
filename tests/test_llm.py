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
    for key in llm._keys:
        key.throttle.interval = 0
    return llm


def rate_limit_error(retry_after="0", message="slow down"):
    request = httpx2.Request("POST", "https://example.com")
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx2.Response(429, request=request, headers=headers)
    return openai.RateLimitError(message, response=response, body=None)


def test_second_identical_call_is_cached():
    llm = make_llm(["feature_request"])
    first = llm.complete("pls add a sleep timer")
    second = llm.complete("pls add a sleep timer")
    assert first.text == second.text == "feature_request"
    assert not first.cached and second.cached
    assert len(llm._keys[0].client.calls) == 1


def test_different_system_prompt_is_a_different_cache_entry():
    llm = make_llm(["a", "b"])
    assert llm.complete("x", system="s1").text == "a"
    assert llm.complete("x", system="s2").text == "b"


def test_retries_on_rate_limit_then_succeeds():
    llm = make_llm([rate_limit_error(), "ok"])
    assert llm.complete("hi").text == "ok"
    assert llm.requests_today() == 2  # failed attempts count too


def test_daily_quota_enforced():
    llm = make_llm(["a", "b"], daily_limit=1)
    llm.complete("one")
    with pytest.raises(QuotaExceeded):
        llm.complete("two")


def test_keys_rotate_and_skip_spent_ones():
    a, b, c = FakeClient(["a1", "a2"]), FakeClient(["b1"]), FakeClient(["c1", "c2"])
    llm = LLM("primary", cache=Cache(":memory:"), client=[a, b, c])
    llm.daily_limit = 1
    for key in llm._keys:
        key.throttle.interval = 0
    llm.cache.exhaust(llm._keys[1].usage_name, 1, llm.quota_tz)
    assert [llm.complete(f"q{i}").text for i in range(2)] == ["a1", "c1"]
    assert len(b.calls) == 0


def test_second_key_takes_over_when_first_hits_quota():
    first, second = FakeClient(["a"]), FakeClient(["b"])
    llm = LLM("primary", cache=Cache(":memory:"), client=[first, second])
    llm.daily_limit = 1
    for key in llm._keys:
        key.throttle.interval = 0
    assert llm.complete("one").text == "a"
    assert llm.complete("two").text == "b"
    assert len(first.calls) == len(second.calls) == 1
    assert llm.requests_today() == 2
    with pytest.raises(QuotaExceeded, match="all 2 keys"):
        llm.complete("three")


def test_provider_daily_cap_moves_to_next_key_without_retrying():
    daily = rate_limit_error(None, "Quota exceeded ... Please retry in 9h28m44.3s.")
    first, second = FakeClient([daily]), FakeClient(["b"])
    llm = LLM("primary", cache=Cache(":memory:"), client=[first, second])
    llm.daily_limit = 1000
    for key in llm._keys:
        key.throttle.interval = 0
    assert llm.complete("one").text == "b"
    assert len(first.calls) == 1  # no backoff retries against the spent key
    assert llm.cache.requests_today(llm._keys[0].usage_name, llm.quota_tz) == 1000


def test_provider_daily_cap_on_last_key_raises_quota_exceeded():
    llm = make_llm([rate_limit_error("40000")], daily_limit=1000)
    with pytest.raises(QuotaExceeded):
        llm.complete("one")


def test_complete_json_reasks_on_bad_json():
    llm = make_llm(["not json at all", '{"label": "bug_report"}'])
    data, resp = llm.complete_json("classify")
    assert data == {"label": "bug_report"}
    assert llm._keys[0].client.calls[0]["response_format"] == {"type": "json_object"}


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


def test_key_env_wildcard_expands_in_numeric_order(monkeypatch):
    from frminer.llm.client import _expand_key_envs

    for i in (1, 2, 10):
        monkeypatch.setenv(f"TESTKEY_{i}", "x")
    assert _expand_key_envs("TESTKEY_*") == ["TESTKEY_1", "TESTKEY_2", "TESTKEY_10"]
    assert _expand_key_envs(["A", "B"]) == ["A", "B"]
