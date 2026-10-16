import pandas as pd
import pytest

from frminer.detect import llm_classify, rules
from frminer.detect.evaluate import bootstrap_metrics, confusion, point_metrics
from frminer.detect.ml import predict, train_tfidf_lr


@pytest.mark.parametrize(
    "text, label",
    [
        ("Please add a sleep timer for podcasts", "feature_request"),
        ("Bring back the old Now Playing screen", "feature_request"),
        ("App keeps crashing when I open my library", "bug_report"),
        ("How do I turn off autoplay?", "question"),
        ("DJ not working, anyone else?", "question"),
        ("Way too many ads, worst app", "complaint"),
        ("Best music app, love it", "praise"),
        ("Spotify Wrapped drops Dec 3", "other"),
        ("it won't let me log in", "bug_report"),
    ],
)
def test_rules(text, label):
    assert rules.classify(text) == label


def test_point_metrics_perfect_and_fr():
    gold = ["feature_request", "complaint", "praise", "feature_request"]
    pred = ["feature_request", "complaint", "praise", "complaint"]
    m = point_metrics(gold, pred)
    assert m["accuracy"] == 0.75
    assert m["fr_precision"] == 1.0
    assert m["fr_recall"] == 0.5
    assert point_metrics(gold, gold)["kappa"] == 1.0


def test_bootstrap_ci_contains_point():
    gold = ["feature_request", "complaint", "praise", "question"] * 10
    pred = ["feature_request", "complaint", "complaint", "question"] * 10
    b = bootstrap_metrics(gold, pred, n_boot=200)
    assert ((b["ci_low"] <= b["value"]) & (b["value"] <= b["ci_high"])).all()
    assert confusion(gold, pred).loc["praise", "complaint"] == 10


def test_llm_prompt_and_validator():
    prompt = llm_classify.build_prompt(["a", "b"])
    assert "[1] a" in prompt and "[2] b" in prompt
    validate = llm_classify._validator(2)
    validate({"labels": {"1": "praise", "2": "other"}})
    with pytest.raises(ValueError):
        validate({"labels": {"1": "praise"}})
    with pytest.raises(ValueError):
        validate({"labels": {"1": "praise", "2": "spam"}})


class FakeResp:
    prompt_tokens, completion_tokens, latency_s, cached = 10, 5, 0.1, False


class FakeLLM:
    def map(self, prompts, **kwargs):
        out = []
        for p in prompts:
            n = p.count("\n[")
            out.append(({"labels": {str(i + 1): "praise" for i in range(n)}}, FakeResp()))
        return out


def test_llm_classify_batches_in_order():
    res = llm_classify.classify(FakeLLM(), [f"t{i}" for i in range(7)], batch_size=3)
    assert res.labels == ["praise"] * 7
    assert res.n_calls == 3


def test_pick_examples_balanced():
    dev = pd.DataFrame({"id": range(12), "text": ["x y"] * 12, "label": ["praise", "other"] * 6})
    ex = llm_classify.pick_examples(dev, per_class=2)
    assert ex["label"].value_counts().to_dict() == {"praise": 2, "other": 2}


def test_tfidf_lr_learns():
    texts = ["please add lyrics", "add a sleep timer pls", "app crashes", "crashes on start"] * 5
    labels = ["feature_request", "feature_request", "bug_report", "bug_report"] * 5
    model = train_tfidf_lr(texts, labels)
    assert list(predict(model, ["please add dark mode", "it crashes"])) == [
        "feature_request",
        "bug_report",
    ]
