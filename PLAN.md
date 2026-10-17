# Project Plan: Feature Request Miner

**A Text Analytics Dashboard for Extracting Customer Feature Requests from Unstructured Product Feedback**

Advanced Social Text project · Solo · ~10 days · Agentic development (Claude Code)

---

## 1. Overview

Users constantly tell companies what they want, but they do it in messy, informal, high-volume text: app store reviews, Reddit threads, tweets. Most of this feedback is *not* a feature request, and the requests that do exist are phrased in hundreds of different, often implicit ways (*"would kill for a sleep timer 💀"*, *"why can't I hide songs??"*, *"Apple Music has lyrics, just saying"*).

This project builds an end-to-end pipeline and dashboard that:

1. **Detects** which feedback items are feature requests (vs. bugs, praise, complaints, questions).
2. **Extracts** the specific feature being requested as a short normalised phrase.
3. **Aggregates** paraphrased requests into coherent themes.
4. **Ranks** themes by volume, engagement, user dissatisfaction and growth.
5. **Presents** everything in an interactive dashboard, including the evaluation of the methods themselves.

**Case study:** Spotify, using Google Play reviews and Reddit (r/truespotify).

The project is **methods-focused**: the core contribution is a rigorous comparison of detection approaches (rules → classical ML → few-shot transformer → LLM) on noisy social text, plus an investigation of LLM-as-annotator distillation.

---

## 2. Research Questions

| # | Question | Answered by |
|---|---|---|
| **RQ1** | How do rule-based, classical ML, few-shot transformer, and LLM approaches compare for classifying feature requests in noisy social text, in terms of accuracy, cost and latency? | Detection experiment (§6.1) |
| **RQ2** | Can LLM-generated "silver" labels train a cheap local classifier that approaches LLM accuracy at a fraction of the cost? | Distillation experiment (§6.2) |
| **RQ3** | How accurately can requested features be extracted and grouped into coherent themes? | Extraction + clustering evaluation (§6.3–6.4) |
| **RQ4** | What do Spotify users request, how does this differ across platforms and over time, and how often do users request features that already exist? | Insights analysis (§6.6) |

---

## 3. Data

### 3.1 Sources

| Source | Tool | Target size | Metadata kept |
|---|---|---|---|
| Google Play reviews (Spotify, `com.spotify.music`) | `google-play-scraper` | full year (~240k raw) | text, star rating, thumbs-up count, date, app version |
| Reddit r/truespotify (posts + top-level comments) | [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) archive API (no key needed) | 5–10k | text, score (upvotes), date, post flair, type (post/comment) |
| *(optional)* Apple App Store reviews | iTunes RSS feed | ≤ 500 per country | text, rating, date |

Time window: 2025-10-16 → 2026-10-16 (12 months).

**Volume vs. LLM budget.** The Play Store alone produces ~650 English reviews/day (~240k/year), far beyond what free LLM tiers can label. Hence the two-tier design: the LLM labels a time-stratified sample (~15k), and the distilled local classifier (RQ2) labels the full corpus, which is what the trend analysis runs on.

> **Why r/truespotify, not r/spotify?** r/spotify has become a playlist-sharing subreddit (its flairs are music genres). App discussion moved to r/truespotify (~50 posts/day), whose flairs (`Feature Request`, `Rant`, `Question`, `News`…) also provide free weak labels to compare against our classifiers.
>
> **Why not the official Reddit API?** Since Reddit's *Responsible Builder Policy*, new API apps require manual approval, which is slow and often denied for personal/academic use. Arctic Shift mirrors Reddit content and exposes a public search API. r/spotify contains heavy playlist-promotion spam, which must be filtered during cleaning (and is itself a social-text noise point for the report).

### 3.1b LLM providers

All free tiers, accessed through OpenAI-compatible endpoints by a single client (`src/frminer/llm/`):

| Profile | Provider | Model | Free limit (observed) |
|---|---|---|---|
| `primary` | Google AI Studio | `gemini-3.5-flash-lite` | generous; ~5–15 s latency |
| `secondary` | Groq | `openai/gpt-oss-20b` | 1,000 req/day, 8k tokens/min |
| `tertiary` | OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` | 50 req/day |

Models are pinned (no auto-routers) and there is no silent fallback between profiles, so every reported result comes from exactly the model it names. Multiple items are batched per request to stay within quotas.

### 3.2 Preprocessing

- Drop empty, non-English (`lingua`) and very short (< 4 words) items. Code-mixed text (e.g. romanised Hindi) is excluded; ~1% of informal English is falsely rejected (documented limitation).
- Deduplicate exact and near-duplicates.
- Keep emojis and casing in the raw text (they carry signal); create a normalised copy for classical models.
- Reddit: split long posts into sentences only for extraction, not for classification.
- **Anonymisation:** no usernames or user IDs are stored. Only text plus aggregate metadata.

### 3.3 Storage

- `data/raw/`: scraped JSONL (gitignored, regenerable).
- `data/interim/corpus.parquet`: full cleaned corpus (gitignored, regenerable with `make data`).
- `data/processed/`: pipeline outputs consumed by the dashboard (committed if small).
- `data/cache/`: LLM response cache, keyed by hash of (model, prompt, input) (gitignored, backed up).
- `annotations/`: gold labels + guidelines (committed; this is irreplaceable human work).

---

## 4. Label Taxonomy

Single-label classification, six classes:

| Label | Definition | Example |
|---|---|---|
| `feature_request` | Asks for new functionality or a change to existing behaviour | "Please add a sleep timer for podcasts" |
| `bug_report` | Something is broken / not working as intended | "App crashes every time I open my library" |
| `complaint` | Negative opinion without a concrete request or bug | "Too many ads, this app is getting worse" |
| `praise` | Positive opinion | "Best music app ever, love the Discover Weekly" |
| `question` | Asks how to do something / seeks information | "How do I download songs on a tablet?" |
| `other` | Spam, off-topic, unintelligible | "first!!" |

**Tie-breaking rules** (encoded in `annotations/guidelines.md`):

- Request + complaint in one item → `feature_request` (the request is the actionable part).
- "Please fix X" where X is broken → `bug_report`; "please change X to work like Y" → `feature_request`.
- Requests to *undo* a change ("bring back the old shuffle") → `feature_request`.
- Implicit requests via comparison ("Apple Music lets you X") → `feature_request`.

---

## 5. Pipeline Architecture

```
          ┌──────────────┐   ┌──────────────┐
          │ Google Play  │   │   Reddit     │
          └──────┬───────┘   └──────┬───────┘
                 └────────┬─────────┘
                          ▼
               ┌────────────────────┐
               │ 1. Ingest + clean  │  → data/processed/corpus.parquet
               └─────────┬──────────┘
                         ▼
               ┌────────────────────┐
               │ 2. Detect (6-way)  │  rules | TF-IDF+LR | SetFit | LLM
               └─────────┬──────────┘
                         ▼  feature_request items only
               ┌────────────────────┐
               │ 3. Extract         │  spaCy patterns | LLM (JSON)
               │    + normalise     │  → 1..n feature phrases per item
               └─────────┬──────────┘
                         ▼
               ┌────────────────────┐
               │ 4. Cluster + name  │  embeddings → HDBSCAN | BERTopic
               └─────────┬──────────┘  LLM writes cluster labels
                         ▼
               ┌────────────────────┐
               │ 5. Rank + trends   │  priority score, emerging themes
               └─────────┬──────────┘
                         ▼
               ┌────────────────────┐
               │ 6. Insights        │  competitors, existing features,
               └─────────┬──────────┘  platform & rating comparison
                         ▼
               ┌────────────────────┐
               │ 7. Dashboard       │  Streamlit (reads processed files)
               └────────────────────┘
```

**Design principles**

- **Every stage is a script** that reads files and writes files, so stages can be re-run independently. `make pipeline` runs everything.
- **All LLM calls are cached on disk**, which keeps the pipeline deterministic on re-run, cheap, and lets the demo run offline.
- **The dashboard never calls heavy models at load time.** It reads precomputed Parquet. Only the "Live Analyser" page runs inference, using the local model by default and the LLM optionally.
- **Config-driven:** model names, thresholds, score weights and paths live in `configs/config.yaml`.

---

## 6. Methods & Experiments

### 6.1 Detection (RQ1): main experiment

| Tier | Method | Training data | Notes |
|---|---|---|---|
| 1 | **Rules**: regex + spaCy matcher (modal/wish patterns, "please add", "why can't", "bring back", comparatives with competitors) | none | transparent baseline; FR vs not-FR, other classes via keyword heuristics |
| 2 | **TF-IDF + Logistic Regression** | gold-dev / silver | classical baseline |
| 3 | **SetFit** (`sentence-transformers` few-shot fine-tuning, CPU) | gold-dev / silver | light-ML transformer |
| 4a | **LLM zero-shot** (OpenRouter) | none | taxonomy definitions in prompt |
| 4b | **LLM few-shot** | k examples from gold-dev | same prompt + examples |
| *opt.* | Second LLM (different provider/size) | none | robustness of LLM findings |

**Metrics:** macro-F1 (6 classes); precision / recall / F1 for `feature_request` specifically; confusion matrices; **cost per 1k items** and **latency per item**; 95% bootstrap confidence intervals on the test set.

### 6.2 LLM-as-annotator distillation (RQ2)

1. LLM (best config from 6.1) labels the **full corpus** → silver labels.
2. Train Tier 2 and Tier 3 on silver labels.
3. Compare on the gold test set: *gold-dev-trained* vs *silver-trained* vs *LLM itself*.
4. Report **Cohen's κ** between LLM and human labels, and accuracy vs cost.
5. *(optional)* Learning curve: silver training size 500 → 1k → 5k → full.

### 6.3 Extraction (RQ3a)

- **Rules:** spaCy dependency patterns, e.g. object of "add/want/need/wish/bring back", noun chunks after "option to / ability to".
- **LLM:** structured JSON output `{"features": ["sleep timer for podcasts"], "competitor": null}`; can return several features per item.
- **Normalisation:** lowercase, lemmatise, strip filler; LLM produces a canonical short form.
- **Evaluation:** manual judgement of ~100 extracted items as *correct / partial / wrong*, for both methods.

### 6.4 Clustering (RQ3b)

- Embed extracted feature phrases (`all-MiniLM-L6-v2` or similar, CPU).
- **Method A:** UMAP + HDBSCAN on phrase embeddings.
- **Method B:** BERTopic on the raw request texts.
- LLM generates a short name + description for each cluster from sample members.
- **Evaluation:** silhouette score / outlier rate (intrinsic) + manual purity check (20 clusters × 10 samples: does each member belong?).

### 6.5 Ranking & trends

**Priority score** (weights adjustable in the dashboard):

```
priority = w1·z(log volume) + w2·z(engagement) + w3·z(share of ≤2-star reviews) + w4·z(growth)
```

- Engagement = thumbs-up (Play) / upvotes (Reddit), normalised per platform.
- Growth = request share in the last N weeks vs the previous N weeks, with minimum-support filtering.
- **Emerging themes:** clusters with significant growth and small but rising volume.

### 6.6 Insights (RQ4)

- **Platform comparison:** request distribution on Play Store vs Reddit.
- **Rating impact:** which requests appear in 1–2★ vs 4–5★ reviews.
- **Competitor mentions:** extract competitor + feature from comparative requests (Apple Music, YouTube Music, Tidal, Deezer, Amazon Music, SoundCloud…).
- **"Already exists" detection:** match request clusters against a curated catalogue of ~40 existing Spotify features (embedding similarity + LLM verification). Requests for existing features signal **discoverability problems**, which is a product insight rather than a missing feature.

---

## 7. Evaluation Data (Gold Set)

- **Size:** 400 items, labelled by hand in the custom labelling app (`make label`).
- **Composition:** 250 Play reviews + 110 Reddit posts + 40 Reddit comments.
- **Sampling** (`src/frminer/annotate.py`): within each source, 50% uniformly random (unbiased estimates) + 50% "enriched", which over-samples by **metadata only** (2–4★ ratings ×2, Feature Request/Rant/Discussion flairs ×3, log length). It never uses text content, so no classifier (rules, ML, LLM) is favoured by how the gold set was built. Items over 250 words are excluded (≈2% of Reddit posts).
- **Split:** **dev = 100** (prompt engineering, rule tuning, few-shot examples, SetFit gold training) / **test = 300** (touched only for final evaluation). Stratified by source × stratum. Dev is labelled first, so method development can start while test is being labelled.
- **Annotation:** one label per item (`annotations/guidelines.md`) + an "unsure" flag + **the requested feature phrase(s)** for feature requests. These phrases double as the **gold set for extraction** (Phase 4).
- **Label provenance:** human labels from scratch, from the text alone (no rating/flair shown, matching what the models see), without seeing any model prediction.
- **Intra-annotator agreement:** the first 27 items were labelled in a pilot pass, discarded, and relabelled from scratch (`annotations/gold_labels_pilot.csv`): κ = 0.76.

---

## 8. Dashboard

**Stack:** Streamlit (multi-page) + Plotly.

| Page | Contents |
|---|---|
| **1. Overview** | KPIs (items analysed, % requests, # themes), top-ranked request themes, priority weight sliders, platform/date filters |
| **2. Request Explorer** | Select a theme → description, volume over time, platform split, rating distribution, representative quotes, competitor mentions |
| **3. Trends** | Emerging vs declining themes, time-series small multiples |
| **4. Insights** | Platform comparison, rating impact, competitor comparison, "already exists" (discoverability) list |
| **5. Model Lab** | Evaluation results: F1 table, confusion matrices, cost vs accuracy scatter, distillation results, error examples |
| **6. Live Analyser** | Paste any text → class + confidence (local SetFit), extracted features and nearest theme (LLM optional) |

There's also a **labelling app** (`apps/labeller.py`), a small Streamlit tool for fast gold annotation (keyboard-friendly, saves after every label, resumable).

---

## 9. Tech Stack

| Concern | Choice |
|---|---|
| Python | 3.12 (managed by `uv`) |
| Env / deps | `uv` + `pyproject.toml` |
| Data | pandas, pyarrow (Parquet) |
| Scraping | `google-play-scraper`, Arctic Shift API (`httpx`) |
| NLP | spaCy (`en_core_web_sm`), `lingua`/`langdetect` |
| Classical ML | scikit-learn |
| Transformers | `sentence-transformers`, `setfit` (CPU) |
| Clustering | `umap-learn`, `hdbscan`, `bertopic` |
| LLM | Gemini, Groq, OpenRouter free tiers (OpenAI-compatible API via `openai` SDK), SQLite response cache |
| Dashboard | Streamlit, Plotly |
| Quality | ruff, pytest, GitHub Actions CI (lint + unit tests) |
| Report | Markdown/LaTeX; figures exported to `reports/figures/` |

---

## 10. Repository Structure

```
social-text-project/
├── PLAN.md
├── README.md
├── pyproject.toml
├── Makefile                    # make setup | scrape | pipeline | dashboard | test
├── .env.example                # OPENROUTER_API_KEY, GEMINI_KEY_1.., GROQ_KEY
├── configs/
│   └── config.yaml             # models, thresholds, weights, paths
├── annotations/
│   ├── guidelines.md
│   ├── gold.csv
│   └── existing_features.yaml  # curated Spotify feature catalogue
├── data/                       # raw/ interim/ cache/ gitignored; processed/ committed
├── src/frminer/
│   ├── config.py
│   ├── ingest/                 # playstore.py, reddit.py, clean.py
│   ├── llm/                    # client.py (OpenRouter + cache), prompts/
│   ├── detect/                 # rules.py, classical.py, setfit_model.py, llm.py
│   ├── extract/                # rules.py, llm.py, normalize.py
│   ├── cluster/                # embed.py, hdbscan_cluster.py, bertopic_cluster.py, naming.py
│   ├── rank/                   # score.py, trends.py
│   ├── insights/               # competitors.py, existing_features.py, compare.py
│   └── evaluate/               # metrics.py, bootstrap.py, error_analysis.py
├── scripts/                    # thin CLI entry points per pipeline stage
├── apps/
│   ├── labeller.py
│   └── dashboard/              # Home.py + pages/
├── notebooks/                  # EDA only, no pipeline logic
├── reports/
│   ├── figures/
│   ├── tables/
│   └── report.md
└── tests/
```

---

## 11. Phased Roadmap

Each phase = one PR merged into `main`. Days are approximate.

### Phase 0: Setup *(Day 1)*
- [x] `uv` project, `pyproject.toml`, Makefile, config, `.env.example`
- [x] Multi-provider LLM client (Gemini / Groq / OpenRouter) with SQLite cache, retries, throttling, daily quota tracking, token + latency logging
- [x] ruff + pytest + GitHub Actions CI
- **Done when:** `make setup && make test` passes; a test LLM call is cached.

### Phase 1: Data collection & EDA *(Day 1–2)*
- [x] Play Store scraper (one resumable stream per star rating)
- [x] Reddit scraper (Arctic Shift, r/truespotify) + removed-post filtering
- [x] Cleaning, language filter, dedup, anonymisation → `corpus.parquet`
- [x] EDA script (`scripts/eda.py`, reproducible, no notebook): volumes over time, length distributions, ratings, flairs, request-cue rates
- **Done when:** full-year clean Play corpus + ≥ 10k Reddit items; EDA figures saved.
- **Result:** 114k clean items: 94k Play reviews, 17.4k Reddit posts, Reddit comments (collection ongoing). 5★ Play reviews were only retrievable for the last ~6 weeks (Google Play pagination limit); 1–4★ cover the full year. Since 5★ reviews carry far fewer request cues (1.9% vs 4–7%), year-long trend analysis uses 1–4★ reviews; rating-mix statistics use the window where all ratings exist.

### Phase 2: Annotation *(Day 2–3)*
- [x] `annotations/guidelines.md` (taxonomy, decision rules, hard cases)
- [x] Gold sampling script (random + metadata-enriched, stratified dev/test split)
- [x] Labelling app (keyboard-driven Streamlit, resumable, feature-phrase capture)
- [x] **👤 You label ~400 items** (~2–3 hrs, can be split across sessions)
- [x] Gold summary (`scripts/gold_summary.py` → `reports/tables/gold_summary.md`)
- **Done when:** `gold.csv` complete with dev/test split; class distribution reported.
- **Result:** 400 items labelled in one ~3 h session. Feature requests are rare: 35 (8.8%; 11 dev / 24 test), 34 with a feature phrase. Complaints dominate (36%); questions are mostly Reddit posts, `other` mostly Reddit comments. Intra-annotator agreement on 27 pilot items labelled before a discarded first pass: 81.5%, κ = 0.76 (most changes were bug ↔ complaint, which was clarified between passes). With only 24 test positives, FR-class F1 has wide CIs; macro-F1 over all six classes is the primary metric and FR metrics are reported with bootstrap CIs.

### Phase 3: Detection & distillation experiments *(Day 3–5)*
- [x] Rule-based classifier (tuned on dev only)
- [x] LLM zero-shot + few-shot (prompt iterated on dev only)
- [x] LLM silver-labels a 12k sample (play 8k, Reddit posts 3k, comments 1k; gold excluded). The rest of the corpus is classified by the distilled model.
- [x] TF-IDF+LR and SetFit trained on gold-dev and on silver
- [x] Evaluation on test: metrics, bootstrap CIs, confusion matrices, cost/latency, κ
- [x] Error analysis: confusion pairs, shared teacher/student mistakes, examples (`make errors`)
- **Done when:** results table + figures in `reports/`; best model chosen for the pipeline.
- **Result** (`reports/tables/detect_test.md`, n=300; dev in `detect_dev.md`):
  - Gemini 3.5 Flash-Lite zero-shot (prompt v2, 20 items per call): macro-F1 0.69 [0.63, 0.74], κ 0.64, ~93 tokens/item. Groq gpt-oss-20b is level (0.69, κ 0.65); few-shot adds nothing overall (FR-F1 0.51 → 0.59, within CI).
  - Prompt v2, which adds two annotator conventions found on dev, beats v1 (0.66 → 0.69, κ 0.59 → 0.64).
  - **Distillation works:** SetFit trained on 11,995 Gemini silver labels matches its teacher (0.69 [0.64, 0.74], κ 0.64, FR-F1 0.58) at ~4 ms/item offline and no API cost. Trained on the 100 gold-dev items instead it reaches only 0.60; TF-IDF+LR goes 0.27 → 0.59 with silver. More silver beyond ~9k did not help on dev.
  - Rules overfit dev (0.63 → 0.48 on test).
  - Main error: the complaint ↔ feature-request / bug boundary (complaint→FR 15, complaint→bug 13 of SetFit's 87 errors); 59% of SetFit's errors are the same wrong label as the teacher's, i.e. likely ambiguous items or annotator conventions.
  - **Deployed:** SetFit (silver), chosen on dev.

### Phase 4: Extraction *(Day 5–6)*
- [ ] Rule-based extractor
- [ ] LLM extractor (JSON, multi-feature, competitor field)
- [ ] Normalisation
- [ ] Evaluation against the gold feature phrases captured during annotation (+ manual check of disagreements)
- **Done when:** `requests.parquet` with features per item; extraction accuracy reported.

### Phase 5: Clustering, ranking & insights *(Day 6–7)*
- [ ] Embedding + HDBSCAN; BERTopic comparison
- [ ] LLM cluster naming
- [ ] Priority score + trend/emerging detection
- [ ] Competitor analysis, existing-feature catalogue + matching, platform & rating comparisons
- [ ] Manual cluster purity check (**👤 ~20 min**)
- **Done when:** `themes.parquet`, `theme_timeseries.parquet`, `insights/*.parquet` produced.

### Phase 6: Dashboard *(Day 7–8)*
- [ ] Six pages per §8
- [ ] Consistent styling, filters, caching (`st.cache_data`)
- [ ] Live Analyser with local model and optional LLM
- **Done when:** `make dashboard` runs the full demo offline (LLM toggle off).

### Phase 7: Report & demo *(Day 9)*
- [ ] All figures/tables generated by script into `reports/`
- [ ] Report draft skeleton with results filled in; **👤 you write the narrative / discussion**
- [ ] Demo script (5–7 min walkthrough)
- **Done when:** report complete; demo rehearsed end-to-end.

### Buffer *(Day 10)*
Polish, optional extensions (App Store data, second LLM, learning curve, intra-annotator κ), demo rehearsal.

---

## 12. Report Outline

1. **Introduction:** problem, motivation, RQs, contributions
2. **Related Work:** app review mining (Maalej & Nabil 2015; Guzman & Maalej 2014; Panichella et al. 2015; Chen et al. AR-Miner 2014), LLMs as annotators (Gilardi et al. 2023), few-shot classification (SetFit, Tunstall et al. 2022), topic modelling for short text (BERTopic, Grootendorst 2022), social-media language models (BERTweet, Nguyen et al. 2020)
3. **Data:** sources, collection, cleaning, statistics, ethics
4. **Annotation:** taxonomy, guidelines, gold set construction, agreement
5. **Methods:** detection tiers, distillation, extraction, clustering, ranking
6. **Experiments & Results:** RQ1–RQ3 tables/figures, error analysis
7. **Insights:** RQ4 findings for Spotify, dashboard walkthrough
8. **Limitations & Ethics:** single annotator, sampling bias, platform ToS, LLM non-determinism, anonymisation
9. **Conclusion & Future Work**

---

## 13. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Reddit API access restricted | **Occurred** | Using Arctic Shift archive API instead; App Store reviews as further fallback |
| Free-tier LLM quotas (OpenRouter 50/day, Groq 8k TPM) | High | Gemini as primary workhorse, batching many items per request, disk cache, local quota tracking; distillation (RQ2) reduces LLM dependence |
| Single annotator = no inter-annotator agreement | Certain | Clear guidelines; optional intra-annotator κ; LLM–human κ; acknowledge in limitations |
| LLM non-determinism | Medium | temperature 0, cached responses, pinned model IDs logged with results |
| Class imbalance (few `question`/`other`) | High | Stratified gold sampling, macro-F1, report per-class |
| Python 3.14 package incompatibility | Medium | Pin Python 3.12 via `uv` |
| Scope creep | Medium | Extensions only after Phase 6 is done |

---

## 14. What I Need From You 👤

| When | What |
|---|---|
| Phase 0 | ✅ API keys in `.env`: `OPENROUTER_API_KEY`, `GEMINI_KEY_1`…`_6`, `GROQ_KEY` (all free tier) |
| Phase 2 | ~2–3 hrs of gold labelling |
| Phase 4–5 | ~1 hr of manual judgement (extraction + clusters) |
| Phase 6 | Feedback on dashboard UX |
| Phase 7 | Report narrative, discussion, demo delivery |

---

## 15. Conventions

- Branch per phase: `phase-N/<short-name>`; PR → squash-merge into `main`.
- Never commit `.env`, raw data or the LLM cache.
- Results that appear in the report must be regenerable by a script (no manual notebook numbers).
- Test-set discipline: **nothing is tuned on gold-test.**
