import json

import numpy as np
import pytest

from frminer.cluster import naming, phrases, purity


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_centroids_and_outlier_assignment():
    emb = unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.8, 0.2], [-1, 0]])
    labels = np.array([0, 0, 1, 1, -1, -1])
    ids, cents = phrases.centroids(emb, labels)
    assert ids.tolist() == [0, 1]
    assert np.allclose(np.linalg.norm(cents, axis=1), 1)
    out = phrases.assign_outliers(emb, labels, threshold=0.6)
    assert out.tolist() == [0, 0, 1, 1, 0, -1]  # [-1, 0] is far from both clusters


def test_hdbscan_finds_two_blobs():
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    labels = phrases.hdbscan_labels(x, min_cluster_size=5, min_samples=2)
    assert len(set(labels) - {-1}) == 2


def test_quality_weights_outliers_by_mentions():
    emb = unit([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [-1, -1]])
    labels = np.array([0, 0, 1, 1, -1])
    q = phrases.quality(emb, labels, weights=np.array([1, 1, 1, 1, 4.0]))
    assert q.n_clusters == 2
    assert q.outlier_rate == pytest.approx(0.2)
    assert q.outlier_mentions == pytest.approx(0.5)
    assert q.silhouette > 0.5
    assert q.largest_share == pytest.approx(0.5)


def test_naming_describe_and_validator():
    assert naming.describe(["b", "a", "c"], [1, 5, 3], top=2) == "a (5); c (3)"
    validate = naming._validator(1)
    validate({"clusters": {"1": {"name": "Fewer Ads", "description": "Less ads."}}})
    with pytest.raises(ValueError):
        validate({"clusters": {"1": {"name": "", "description": "x"}}})


def test_purity_prompt_and_validator():
    item = json.dumps({"label": "Fewer Ads", "members": ["fewer ads", "lyrics"]})
    prompt = purity.build_prompt([item])
    assert "Label: Fewer Ads" in prompt and "2. lyrics" in prompt
    validate = purity._validator([2])(1)
    validate({"clusters": {"1": {"1": True, "2": False}}})
    with pytest.raises(ValueError):
        validate({"clusters": {"1": {"1": "yes"}}})


class FakeResp:
    prompt_tokens, completion_tokens, latency_s, cached = 10, 5, 0.1, False


class FakeLLM:
    """Says the first member belongs and omits the rest."""

    def map(self, prompts, **kwargs):
        out = []
        for p in prompts:
            n = p.count("\n[")
            out.append(({"clusters": {str(i + 1): {"1": True} for i in range(n)}}, FakeResp()))
        return out


def test_purity_judge_marks_missing_members_none():
    res = purity.judge(FakeLLM(), ["a", "b"], [["x", "y"], ["z"]], batch_size=2)
    assert res.outputs == [[True, None], [True]]


def test_merge_validator_and_apply():
    from frminer.cluster import merge

    v = merge._validator({1, 2, 3})
    v({"groups": [[1, 2]]})
    with pytest.raises(ValueError):
        v({"groups": [[1, 2], [2, 3]]})
    with pytest.raises(ValueError):
        v({"groups": [[1, 9]]})
    out = merge.apply_groups(np.array([3, 1, 2, -1]), [[2, 3]])
    assert out.tolist() == [2, 1, 2, -1]


class VerifyLLM:
    """Confirms a pair only when both descriptions mention 'skip'."""

    def map(self, prompts, **kwargs):
        out = []
        for p in prompts:
            blocks = p.split("Items:\n\n")[1].split("\n\n")
            same = {
                str(i + 1): all("skip" in line for line in b.splitlines()[1:])
                for i, b in enumerate(blocks)
            }
            out.append(({"same": same}, FakeResp()))
        return out


def test_verify_groups_keeps_only_confirmed_members():
    from frminer.cluster import merge

    descs = {1: "remove skip limit (90)", 2: "unlimited skips (40)", 3: "clear queue (5)"}
    groups, checked = merge.verify_groups(VerifyLLM(), [[1, 2, 3]], descs, {1: 90, 2: 40, 3: 5})
    assert groups == [[1, 2]]
    assert sorted(checked) == [(1, 2, True), (1, 3, False)]
