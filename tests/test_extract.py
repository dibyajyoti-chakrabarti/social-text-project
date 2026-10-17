import json

import numpy as np
import pytest

from frminer.extract import judge, llm_extract, rules
from frminer.extract.evaluate import (
    best_token_f1,
    item_counts,
    matches,
    scores,
    split_gold,
)
from frminer.extract.normalize import clean, normalize


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("Please add an option to hide songs!", "hide songs"),
        ("I wish I could see the lyrics", "see lyrics"),
        ("sleep timer for podcasts 💤", "sleep timer for podcasts"),
        ("more", ""),
    ],
)
def test_clean(phrase, expected):
    assert clean(phrase) == expected


def test_normalize_lemmatises():
    assert normalize("Add lyrics to podcasts") == "lyric to podcast"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Please add a sleep timer for podcasts!", ["sleep timer for podcasts"]),
        ("I wish I could hide songs from my playlists", ["hide songs from my playlists"]),
        ("Bring back the old shuffle. It was better", ["bring back old shuffle"]),
        ("Give us an option to disable autoplay", ["disable autoplay"]),
        ("You should add a dark mode", ["dark mode"]),
        ("I add about ten songs to my queue", []),
        ("too many ads", []),
    ],
)
def test_rule_extractor(text, expected):
    assert rules.extract(text) == expected


def test_competitors():
    assert rules.competitors("Apple Music has lyrics and YT Music is free") == [
        "Apple Music",
        "YouTube Music",
    ]
    assert rules.competitors("tidal wave of ads") == ["Tidal"]  # keyword lists are crude


def test_llm_extract_validator_and_parse():
    validate = llm_extract._validator(2)
    good = {
        "items": {
            "1": {"features": ["Sleep Timer", "sleep timer", " "], "competitor": None},
            "2": {"features": [], "competitor": "Apple Music"},
        }
    }
    validate(good)
    assert llm_extract._parse(good, 0) == llm_extract.Extraction(["sleep timer"], None)
    assert llm_extract._parse(good, 1).competitor == "Apple Music"
    with pytest.raises(ValueError):
        validate({"items": {"1": {"features": "lyrics"}, "2": {"features": []}}})
    with pytest.raises(ValueError):
        validate({"items": {"1": {"features": []}}})


class FakeResp:
    prompt_tokens, completion_tokens, latency_s, cached = 10, 5, 0.1, False


class FakeLLM:
    def map(self, prompts, **kwargs):
        out = []
        for p in prompts:
            n = p.count("\n[")
            items = {str(i + 1): {"features": ["lyrics"], "competitor": None} for i in range(n)}
            out.append(({"items": items}, FakeResp()))
        return out


def test_llm_extract_batches():
    res = llm_extract.extract(FakeLLM(), ["a", "b", "c"], batch_size=2)
    assert [e.features for e in res.outputs] == [["lyrics"]] * 3
    assert res.n_calls == 2


def test_judge_prompt_and_validator():
    items = [json.dumps({"text": "add lyrics", "extracted": ["lyrics"], "reference": None})]
    prompt = judge.build_prompt(items)
    assert "[1]" in prompt and 'Extracted: ["lyrics"]' in prompt and "Reference" not in prompt
    judge._validator(1)({"grades": {"1": "partial"}})
    with pytest.raises(ValueError):
        judge._validator(1)({"grades": {"1": "meh"}})


def test_split_gold():
    assert split_gold("sleep timer; lyrics ;") == ["sleep timer", "lyrics"]
    assert split_gold(None) == []


def test_matching_is_one_to_one_and_optimal():
    sim = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert matches(sim, 0.6) == 2  # greedy (p0-g0 first) would find only 1
    assert matches(np.array([[0.9, 0.95]]), 0.6) == 1
    assert matches(np.array([[0.3]]), 0.6) == 0


def fake_embed(phrases):
    """Bag-of-words vectors: phrases sharing words are similar."""
    vocab = sorted({w for p in phrases for w in p.split()})
    vecs = np.array([[p.split().count(w) for w in vocab] for p in phrases], dtype=float)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_item_counts_and_scores():
    preds = [["sleep timer", "dark mode"], [], ["lyrics"]]
    golds = [["sleep timer for podcasts"], ["fewer ads"], ["lyrics"]]
    counts = item_counts(preds, golds, fake_embed, threshold=0.6)
    assert counts["matched"].tolist() == [1, 0, 1]
    s = scores(counts)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["empty"] == pytest.approx(1 / 3) and s["item_hit"] == pytest.approx(2 / 3)


def test_best_token_f1():
    assert best_token_f1([["sleep timer"]], [["sleep timer"]]) == 1.0
    assert best_token_f1([[]], [["lyrics"]]) == 0.0
