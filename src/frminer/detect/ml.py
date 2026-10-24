"""Tiers 2–3: TF-IDF + a linear classifier (logistic regression, linear SVM or complement
naive Bayes), and SetFit (few-shot sentence-transformer).

Both train on either the 100 gold-dev items or the LLM silver labels (RQ2).
torch / setfit are imported lazily so the rest of the package works without the
optional `ml` dependency group (e.g. in CI).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from frminer.config import path

SETFIT_BASE = "sentence-transformers/all-mpnet-base-v2"
SETFIT_MAX_TOKENS = 256  # fits the 8 GB laptop GPU; covers ~97% of gold items untruncated


def tfidf_features(min_df: int = 2) -> FeatureUnion:
    """Word 1–2-grams + character 2–5-grams, shared by every TF-IDF model."""
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), min_df=min_df, sublinear_tf=True
                ),
            ),
        ]
    )


def tfidf_lr(min_df: int = 2, C: float = 4.0) -> Pipeline:
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", tfidf_features(min_df)), ("clf", clf)])


def tfidf_svm(min_df: int = 2, C: float = 0.03) -> Pipeline:
    """C chosen on gold dev (scripts/detect_dev.py, SVM_C grid)."""
    clf = LinearSVC(C=C, class_weight="balanced")
    return Pipeline([("features", tfidf_features(min_df)), ("clf", clf)])


def tfidf_nb(min_df: int = 2, alpha: float = 0.03) -> Pipeline:
    """Complement NB (copes with imbalanced classes); alpha chosen on gold dev (NB_ALPHA grid)."""
    return Pipeline([("features", tfidf_features(min_df)), ("clf", ComplementNB(alpha=alpha))])


def _train_tfidf(make, texts: Sequence[str], labels: Sequence[str], **kwargs) -> Pipeline:
    min_df = kwargs.pop("min_df", 1 if len(texts) < 1000 else 2)
    return make(min_df=min_df, **kwargs).fit(list(texts), list(labels))


def train_tfidf_lr(texts: Sequence[str], labels: Sequence[str], **kwargs) -> Pipeline:
    return _train_tfidf(tfidf_lr, texts, labels, **kwargs)


def train_tfidf_svm(texts: Sequence[str], labels: Sequence[str], **kwargs) -> Pipeline:
    return _train_tfidf(tfidf_svm, texts, labels, **kwargs)


def train_tfidf_nb(texts: Sequence[str], labels: Sequence[str], **kwargs) -> Pipeline:
    return _train_tfidf(tfidf_nb, texts, labels, **kwargs)


def train_setfit(
    texts: Sequence[str],
    labels: Sequence[str],
    *,
    base: str = SETFIT_BASE,
    num_iterations: int = 20,
    max_steps: int = -1,
    batch_size: int = 8,
    seed: int = 0,
    out_dir: Path | None = None,
):
    """Contrastive fine-tuning of the sentence encoder + logistic-regression head."""
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(
        base, head_params={"class_weight": "balanced", "max_iter": 3000}
    )
    model.model_body.max_seq_length = SETFIT_MAX_TOKENS
    args = TrainingArguments(
        batch_size=batch_size,
        num_iterations=num_iterations,
        max_steps=max_steps,
        num_epochs=1,
        seed=seed,
        report_to="none",
        show_progress_bar=True,
        save_strategy="no",  # no intermediate checkpoints
        output_dir=str(path("cache") / "setfit"),
    )
    train = Dataset.from_dict({"text": list(texts), "label": list(labels)})
    Trainer(model=model, args=args, train_dataset=train).train()
    if out_dir is not None:
        model.save_pretrained(str(out_dir))
    return model


def load_setfit(model_dir: Path):
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(model_dir))
    model.model_body.max_seq_length = SETFIT_MAX_TOKENS
    return model


def predict_proba(model, texts: Sequence[str], batch_size: int = 64) -> tuple[np.ndarray, list]:
    """Class probabilities (rows = texts) and the class order of the columns."""
    texts = list(texts)
    if hasattr(model, "model_body"):
        proba = model.predict_proba(texts, batch_size=batch_size, show_progress_bar=True)
        proba = proba.cpu().numpy() if hasattr(proba, "cpu") else np.asarray(proba)
        return proba, list(model.model_head.classes_)
    return model.predict_proba(texts), list(model.classes_)


def predict(model, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    """Predict with either a sklearn pipeline or a SetFit model."""
    texts = list(texts)
    if not hasattr(model, "model_body"):  # sklearn pipeline (LinearSVC has no predict_proba)
        return model.predict(texts)
    return np.asarray(model.predict(texts, batch_size=batch_size, show_progress_bar=False))
