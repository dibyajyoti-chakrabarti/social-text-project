"""Package the corpus and gold labels as a Kaggle dataset → data/kaggle/.

uv run python scripts/export_kaggle.py
kaggle datasets create -p data/kaggle            # first upload (private)
kaggle datasets version -p data/kaggle -m "..."  # later updates

IDs are already anonymised (clean.anon_id), so the export only reshapes and documents.
"""

import json
import shutil

import pandas as pd

from frminer.annotate import load_gold
from frminer.config import ROOT, load_config, path

DATASET_ID = "dibyajyoti122/spotify-feature-requests"

SUBTITLE = "117,875 Play Store reviews and Reddit posts, with 400 hand-labelled items"

DESCRIPTION = """\
One year of public Spotify user feedback from Google Play reviews and the r/truespotify \
subreddit, collected for a study of feature-request mining in noisy social text.

## Files

- **corpus.csv**: the full cleaned corpus ({n_corpus:,} items, {since} to {until}).
- **gold_labels.csv**: {n_gold} items labelled by hand with one of six intents \
(feature_request, bug_report, complaint, praise, question, other), plus the requested \
feature phrases for feature requests. Split into dev ({n_dev}) and test ({n_test}).
- **annotation_guidelines.md**: the labelling rules used for the gold set.

## Collection and cleaning

- Play Store: `google-play-scraper`, app `com.spotify.music`, English, US store.
- Reddit: posts and top-level comments from r/truespotify via the Arctic Shift archive API.
- Dropped empty, non-English and very short (under {min_words} words) items, and \
duplicates (same ID, or same text ignoring case and punctuation; the most engaged copy is kept).
- No usernames are kept. Every ID is a truncated SHA-1 hash of the original ID.

## Known caveats

- 5-star Play reviews are only present from 2026-08-31 (a Google Play pagination limit). \
1 to 4 star reviews cover the full window. Leave 5-star reviews out of trend analysis.
- The gold set has a single annotator (intra-annotator Cohen's kappa 0.76).
- Half of the gold items are a random sample; the other half are an enriched sample that \
over-weights longer texts, 2 to 4 star ratings and request-like flairs, so class shares in \
the gold set are not corpus shares.

## Licence

Labels and packaging are CC BY-NC-SA 4.0. The review and post text belongs to its \
original authors and is shared for non-commercial research only.
"""

CORPUS_FIELDS = [
    ("id", "string", "Anonymised ID; prefix play_, rp_ (Reddit post) or rc_ (Reddit comment)"),
    ("source", "string", "play, reddit_post or reddit_comment"),
    ("text", "string", "Raw text with emojis and casing kept; Reddit posts are title + body"),
    ("created_at", "datetime", "Posting time, UTC"),
    ("rating", "integer", "Play Store star rating 1 to 5; empty for Reddit"),
    ("engagement", "integer", "Play thumbs-up count, or Reddit score"),
    ("flair", "string", "Reddit post flair (e.g. Feature Request, Rant); empty for Play"),
    ("app_version", "string", "App version reported with the Play review"),
    ("parent_id", "string", "Reddit comments: anonymised ID of the post they reply to"),
    ("n_words", "integer", "Whitespace word count of text"),
]

GOLD_FIELDS = [
    ("id", "string", "Anonymised ID, joins to corpus.csv"),
    ("split", "string", "dev (used for prompt and model development) or test (held out)"),
    ("stratum", "string", "random, or enriched (sampled with request-likely metadata weights)"),
    ("source", "string", "play, reddit_post or reddit_comment"),
    ("rating", "integer", "Play Store star rating; empty for Reddit"),
    ("flair", "string", "Reddit post flair; empty for Play"),
    ("created_at", "datetime", "Posting time, UTC"),
    ("text", "string", "Item text as shown to the annotator"),
    ("label", "string", "Gold intent label (see annotation_guidelines.md)"),
    ("features", "string", "Requested features, ';'-separated; feature_request items only"),
    ("unsure", "boolean", "Annotator hesitated between two labels"),
]


def schema(fields: list[tuple[str, str, str]]) -> dict:
    return {"fields": [{"name": n, "type": t, "description": d} for n, t, d in fields]}


def main() -> None:
    out = ROOT / "data" / "kaggle"
    out.mkdir(parents=True, exist_ok=True)

    corpus = pd.read_parquet(path("interim") / "corpus.parquet")
    corpus = corpus[[name for name, _, _ in CORPUS_FIELDS]]
    corpus.to_csv(out / "corpus.csv", index=False)

    gold = load_gold(path("annotations"))
    gold["rating"] = gold["rating"].astype("Int64")
    gold["unsure"] = gold["unsure"].astype(bool)
    gold = gold[[name for name, _, _ in GOLD_FIELDS]]
    gold.to_csv(out / "gold_labels.csv", index=False)

    shutil.copy(path("annotations") / "guidelines.md", out / "annotation_guidelines.md")

    data_cfg = load_config()["data"]
    until = pd.Timestamp(data_cfg["until"]) - pd.Timedelta(days=1)  # config bound is exclusive
    meta = {
        "title": "Spotify Feature Requests: Play and Reddit",
        "subtitle": SUBTITLE,
        "id": DATASET_ID,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "keywords": ["text", "nlp", "classification", "music"],
        "description": DESCRIPTION.format(
            n_corpus=len(corpus),
            since=data_cfg["since"],
            until=until.date(),
            n_gold=len(gold),
            n_dev=(gold["split"] == "dev").sum(),
            n_test=(gold["split"] == "test").sum(),
            min_words=data_cfg["min_words"],
        ),
        "resources": [
            {
                "path": "corpus.csv",
                "description": "Full cleaned corpus",
                "schema": schema(CORPUS_FIELDS),
            },
            {
                "path": "gold_labels.csv",
                "description": "Hand-labelled gold sample",
                "schema": schema(GOLD_FIELDS),
            },
            {"path": "annotation_guidelines.md", "description": "Labelling rules for the gold set"},
        ],
    }
    with open(out / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(corpus)} corpus rows and {len(gold)} gold rows to {out}")


if __name__ == "__main__":
    main()
