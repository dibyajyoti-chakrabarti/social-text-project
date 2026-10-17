"""Classify the whole corpus with the deployed local model → data/processed/classified.parquet.

uv run python scripts/classify_corpus.py [--retrain]

The deployed model (configs: detect.deployed) is SetFit trained on the LLM silver labels,
chosen on dev in Phase 3. It is trained once and saved under models/; later runs load it.
Output columns: id, label, confidence (probability of the predicted class).
"""

import sys

import numpy as np
import pandas as pd

from frminer.config import load_config, path
from frminer.detect import ml


def main() -> None:
    cfg = load_config()["detect"]["deployed"]
    model_dir = path("models") / f"{cfg['method']}_{cfg['train']}"
    if (model_dir / "config.json").exists() and "--retrain" not in sys.argv:
        model = ml.load_setfit(model_dir)
    else:
        silver = pd.read_parquet(path("processed") / "silver.parquet")
        model = ml.train_setfit(
            silver["text"],
            silver["label"],
            max_steps=cfg["max_steps"],
            seed=cfg["seed"],
            out_dir=model_dir,
        )

    corpus = pd.read_parquet(path("interim") / "corpus.parquet")[["id", "source", "text"]]
    proba, classes = ml.predict_proba(model, corpus["text"], batch_size=128)
    out = pd.DataFrame(
        {
            "id": corpus["id"],
            "label": np.asarray(classes)[proba.argmax(axis=1)],
            "confidence": proba.max(axis=1).round(4),
        }
    )
    out.to_parquet(path("processed") / "classified.parquet", index=False)
    print(pd.crosstab(out["label"], corpus["source"], margins=True).to_string())


if __name__ == "__main__":
    main()
