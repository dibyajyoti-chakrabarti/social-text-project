<!--
How this file works: `make report` renders this template into reports/report.md.
Every number is a double-brace field read from reports/tables (written by the pipeline scripts),
so the report always matches the code. Write prose here and never type a result by hand.
-->

# Feature Request Miner: Extracting Customer Feature Requests from Unstructured Product Feedback

Advanced Social Text project · *authors* · October 2026

## Abstract

Product teams receive far more user feedback than they can read, and the feature requests in it are a
small, informally phrased minority. We present *Feature Request Miner*, a pipeline and dashboard that
detects feature requests in noisy social text, extracts the requested feature as a short phrase, groups
paraphrased requests into themes and ranks them. As a case study we collected one year of Spotify feedback,
{{ n(corpus['english']) }} English Google Play reviews and r/truespotify posts and comments, and labelled a
gold set of {{ gold['items'] }} items by hand. On the held-out test split an LLM classifier reaches
macro-F1 {{ f2(teacher['macro_f1']) }}; a SetFit model trained on {{ n(det.loc['SetFit (silver)', 'train'].split()[0]) }}
LLM-generated (silver) labels matches it ({{ f2(dep['macro_f1']) }}) while running locally, about
{{ int(round(teacher['s_per_item'] / dep['s_per_item'])) }}× faster per item and without API cost; hand-written
rules reach {{ f2(det.loc['Rules', 'macro_f1']) }}. LLM extraction is graded correct for
{{ pct(ext.loc[('gold all', 'llm'), 'judge_correct']) }} of gold requests, against
{{ pct(ext.loc[('gold all', 'rules'), 'judge_correct']) }} for a dependency-rule baseline. Clustering
{{ n(grid['mentions']) }} extracted mentions yields {{ grid['themes'] }} themes with a human-judged purity of
{{ f2(pur.loc['A human', 'purity']) }}. Free-tier limits dominate what users ask for (the top themes are
{{ names(top, 3) }}), opt-outs from AI and recommendation features appear in the angriest reviews, and
{{ ins['status']['exists'] }} themes ask for features Spotify already offers to the users asking. Every number is
written by a script, and the results can be explored in a six-page interactive dashboard.

## 1. Introduction

App stores and online communities give software companies a continuous stream of unsolicited feedback. Our
one-year sample for a single app, Spotify on Android, contains {{ n(corpus['english']) }} usable English items,
far more than a product team can read. Most of it is not actionable: in our hand-labelled sample only
{{ pct(gold['by_split'].loc['feature_request', 'share'], 1) }} of items are feature requests, while complaints make
up {{ pct(gold['by_split'].loc['complaint', 'share'], 1) }}. The requests that do exist are phrased informally and often
implicitly (*"would kill for a sleep timer"*, *"why can't I hide songs??"*, *"Apple Music has lyrics, just
saying"*), so keyword search misses many of them and returns many false positives, and the same request is
written in hundreds of different ways.

We chose Spotify because it is a large consumer app with a freemium model, so its feedback mixes functional
wishes (a sleep timer, better shuffle) with business requests (fewer ads, more skips, cheaper plans), and
because its users discuss it actively in public. We combine two channels that reach different users. Google
Play reviews are short, carry a star rating, and come from a broad population that includes many free-tier
users. The r/truespotify subreddit is an enthusiast community that writes longer posts, replies to each other
and votes on posts. Comparing the two shows how much the picture of "what users want" depends on where one
listens.

**Research questions**

- **RQ1** How accurately can feature requests be detected in app reviews and Reddit posts, and how do rules, LLMs and small local models compare in accuracy and cost?
- **RQ2** Can a small local model trained on LLM-generated (silver) labels match the LLM?
- **RQ3** Can the requested features be extracted (RQ3a) and grouped into coherent themes (RQ3b)?
- **RQ4** What do Spotify users ask for, how does it differ by platform and rating, and which requests concern features that already exist?

**Contributions**

- A reproducible end-to-end pipeline, from scraping to a ranked list of request themes, applied to one year of
  Spotify feedback from two platforms. Every stage is a script, and every number in this report is read from a
  table those scripts write.
- A comparison of rules, two LLM families, TF-IDF + logistic regression and SetFit for six-way feedback
  classification on a held-out gold test set, including latency, and a distillation experiment showing that a
  local model trained on LLM labels matches its LLM teacher.
- An evaluation of feature extraction and clustering in which the LLM judge is calibrated against a human check,
  showing that LLM-judged cluster purity overstates human-judged purity.
- Insights for Spotify, including which requests concern features that already exist (a discoverability
  problem) or exist only in Premium (a paywall signal), delivered through an interactive dashboard with a live
  analyser for new text.

## 2. Related Work

**App review mining.** Mining app store reviews for software engineering is an established field.
AR-Miner (Chen et al., 2014) filtered out non-informative reviews, grouped the rest with topic models and ranked
the groups for developers, which is the closest predecessor of our pipeline. Guzman and Maalej (2014) extracted
fine-grained app features from reviews with collocation finding and attached sentiment to them. Maalej and Nabil
(2015) classified reviews into bug reports, feature requests, user experiences and ratings, and showed that
metadata such as the star rating helps text classifiers. Panichella et al. (2015) combined linguistic patterns,
text features and sentiment to sort reviews into maintenance categories such as feature request and problem
discovery. Our label set follows these taxonomies, with *question* and *other* added because Reddit
conversations contain many of both. These works used app store data only and trained supervised models on
thousands of manual labels; we add Reddit and replace most manual labelling with LLM labels.

**LLMs as annotators.** Gilardi et al. (2023) found that zero-shot ChatGPT outperformed crowd workers on several
text annotation tasks at a fraction of the cost. Wang et al. (2021) used GPT-3 to label training data for smaller
models and reduced labelling cost substantially at similar accuracy. Training a small student on a larger
teacher's outputs is knowledge distillation (Hinton et al., 2015); we use the simplest form, training on the
teacher's hard labels.

**Few-shot classification.** SetFit (Tunstall et al., 2022) fine-tunes a Sentence-BERT encoder (Reimers and
Gurevych, 2019) contrastively on pairs of labelled examples and then trains a light classification head. It is
competitive with much larger models when only a few labels exist, which makes it a natural student for
distillation and a natural baseline for our 100 gold-dev labels.

**Topic modelling for short text.** Classical topic models such as LDA struggle with short, sparse documents.
BERTopic (Grootendorst, 2022) instead clusters document embeddings, reducing them with UMAP (McInnes et al., 2018)
and clustering them with HDBSCAN (Campello et al., 2013), then describes each cluster with class-based TF-IDF. We
use the same components but cluster the extracted feature phrases instead of whole documents, and use BERTopic on
the full texts as the comparison.

**Language models for social media.** BERTweet (Nguyen et al., 2020) showed that pre-training on tweets helps on
noisy social text. We did not pre-train a domain model; we use general-purpose sentence encoders and LLMs, and
leave domain-adapted encoders to future work.

## 3. Data

We collected Spotify Android reviews from Google Play and posts and comments from r/truespotify
(via the Arctic Shift archive). After de-duplication, a minimum-length filter and language identification,
{{ n(corpus['raw']) }} raw items became **{{ n(corpus['english']) }}** English items:

| step | items |
|:--|--:|
| collected | {{ n(corpus['raw']) }} |
| after id de-duplication | {{ n(corpus['after_id_dedup']) }} |
| after minimum length | {{ n(corpus['min_words']) }} |
| after text de-duplication | {{ n(corpus['after_text_dedup']) }} |
| English | {{ n(corpus['english']) }} |

{{ md(csum.reset_index(), pct_cols=('engagement > 0', 'explicit request cue')) }}

{{ corpus_note }}

![Weekly volume by source](figures/eda_weekly_volume.png)

![Play Store rating distribution](figures/eda_ratings.png)

**Ethics.** We collected only publicly visible content: Play Store reviews from the public store listing and
Reddit content from the Arctic Shift archive, which mirrors public Reddit data. Posts removed by moderators and
removed or deleted comments were dropped; for posts whose author deleted the body, only the title was kept. Usernames, avatars and author names were never written to disk;
items carry anonymised ids, and mentions of Reddit users inside the text are masked. The corpus keeps only text,
date, rating, engagement, app version and (for Reddit) flair. Quotes in this report and the dashboard are short,
are not linked to accounts, and we made no attempt to identify anyone. The text was sent to third-party LLM
APIs (Google Gemini and Groq) for labelling and extraction; it is public text, but this is a further copy outside
the platforms. Google Play's terms do not invite automated collection; we scraped at a low rate, for
non-commercial academic research only, and neither the raw data nor the LLM response cache is committed to
the repository or redistributed.

## 4. Annotation

We defined six labels (feature_request, bug_report, complaint, praise, question, other) and wrote
guidelines (`annotations/guidelines.md`). A stratified gold sample of **{{ gold['items'] }}** items
(enriched for request cues plus random) was labelled by one annotator and split into
{{ gold['by_split'].loc['total', 'dev'] }} dev and {{ gold['by_split'].loc['total', 'test'] }} test items.
The test split was used only for final scoring.

{{ md(gold['by_split'].reset_index(), pct_cols=('share',)) }}

Intra-annotator agreement (a re-labelling pass over {{ gold['pilot_items'] }} pilot items):
raw agreement {{ pct(gold['agreement'], 1) }}, Cohen's κ = **{{ f2(gold['kappa']) }}**.

**Hard boundaries.** Three boundaries caused most of the difficulty. *Feature request vs complaint*: much
negative feedback implies a wish ("way too many ads now"). The guidelines require an actionable ask: "way too
many ads" is a complaint, "please reduce the ads or let us skip them" is a request. Requests about ads, pricing
and plans count as feature requests, because they are changes users ask for. *Bug report vs complaint*: a
malfunction ("lyrics disappeared after the update") is a bug, whereas a deliberate change the user dislikes is a
complaint, or a request if they ask for it to be reversed. Most of the label changes between the two pilot
passes were on this boundary, and it was clarified in the guidelines between them. *Question vs other*, mostly
in Reddit comments: replies often end in a question to the poster, and only questions that seek information
count. Mixed items take the first applicable label in the priority order feature_request > bug_report >
question > complaint > praise > other.

**Conventions inferred from dev.** Comparing the first LLM prompt (v1, written from the guidelines) with the
dev labels showed two conventions the annotator applied consistently but the guidelines did not state
explicitly: (1) describing a disliked behaviour ("they removed X", "I don't like how it plays random songs") is a
complaint unless there is an explicit ask; (2) a problem description that asks other users for help or
confirmation ("anyone else?", "any fix?") is a question, not a bug report. Prompt v2 adds both. They were found
on dev only; the test labels were never changed after scoring.

## 5. Methods

The pipeline has five stages, each a script that reads the previous stage's output from disk. All LLM calls use
temperature 0 and are cached in a local SQLite database, so every stage can be re-run without new API calls.
All design choices (rules, prompts, hyperparameters and the model to deploy) were made on gold-dev.

- **Detection (RQ1).** Regex rules; Gemini ({{ teacher['model'] }}) zero- and few-shot with two prompt versions;
  a second LLM family (Groq {{ det.loc['LLM-2 zero-shot', 'model'] }}); TF-IDF + logistic regression and SetFit
  (all-mpnet-base-v2) trained on either the {{ gold['by_split'].loc['total', 'dev'] }} gold-dev items or
  {{ n(det.loc['SetFit (silver)', 'train'].split()[0]) }} Gemini silver labels. The LLMs classify 20 items per
  call and return JSON. The silver sample is drawn from the whole corpus with gold items excluded.
- **Distillation (RQ2).** The SetFit model trained on silver labels is deployed on the full corpus.
  It was chosen on dev, where it scored {{ f3(dev.loc['setfit_silver', 'macro_f1']) }} against
  {{ f3(dev.loc['llm_v2_zero_b20', 'macro_f1']) }} for its teacher, and it needs no API calls.
- **Extraction (RQ3a).** Gemini extracts short canonical feature phrases plus a cited competitor;
  a dependency-parse rule extractor is the baseline. A judge from a different family (Groq) grades both.
  The LLM may return several features per item, or none when the item is not actually a request. The rules match
  spaCy dependency patterns such as *please add X*, *bring back X* and *I wish X* plus a list of competitor names.
  Phrases are normalised (fillers and articles stripped, nouns lemmatised). The judge sees only the text (and, for
  gold items, the annotator's phrases), not which method produced an extraction.
- **Clustering (RQ3b).** {{ n(grid['phrases']) }} distinct phrases ({{ n(grid['mentions']) }} mentions) embedded with
  MiniLM, reduced with UMAP and clustered with HDBSCAN over a grid of {{ grid['settings'] }} settings,
  chosen by rule (min_cluster_size={{ grid['min_cluster_size'] }}, min_samples={{ grid['min_samples'] }},
  {{ grid['clusters_before'] }} clusters). Gemini proposed {{ grid['proposed_merges'] }} merges, Groq confirmed
  {{ grid['confirmed'] }}, leaving **{{ grid['themes'] }} themes**. BERTopic on item texts is the comparison.
  The selection rule takes the highest silhouette among settings that leave few mentions unassigned and give
  a manageable number of clusters, so no setting is picked by eye. Noise phrases close to a cluster centroid are
  attached to it afterwards. Gemini then names each theme and writes a one-line description.
- **Ranking.** priority = 0.4·z(log items) + 0.2·z(engagement) + 0.2·z(share ≤2★) + 0.2·z(growth);
  trends compare the last {{ trends['weeks'] }} weeks with the {{ trends['weeks'] }} before (two-proportion test, BH correction).
  Engagement is each item's percentile of thumbs-up or upvotes within its own source, so Play and Reddit are
  comparable. Growth is the log ratio of the theme's share of all corpus items in the two windows, which removes
  changes in how much feedback was collected. 5★ Play reviews are left out of growth, trends and monthly
  shares, because Google Play only returns them for the last weeks of the year; counting them would make
  every theme appear to shrink in the recent window. The weights are a judgement call; the dashboard lets users change
  them and see how the ranking moves.
- **Insights.** Platform lean compares a theme's share of Play items with its share of Reddit items. For the
  "already exists" analysis we wrote a catalogue of current Spotify features with their availability (everyone or
  Premium only); Gemini matches each theme against it and assigns one of four statuses: *exists*, *premium_only*,
  *partly* (a limited version exists) or *new*.

## 6. Experiments and Results

### 6.1 Detection (RQ1)

All systems on the gold test split (n = {{ int(dep['n']) }}), 95% bootstrap CIs:

{{ md(det.reset_index(), ['system', 'train', 'macro_f1', 'macro_f1_ci', 'accuracy', 'kappa', 'fr_f1', 'fr_f1_ci', 's_per_item']) }}

The best LLM setting reaches macro-F1 {{ f3(det[det.index.str.startswith('LLM')]['macro_f1'].max()) }}; the deployed SetFit (silver) model
scores {{ f3(dep['macro_f1']) }} {{ dep['macro_f1_ci'] }}, against {{ f3(teacher['macro_f1']) }} for its teacher,
at {{ f3(dep['s_per_item']) }} s per item instead of {{ f3(teacher['s_per_item']) }}.
Rules reach {{ f3(det.loc['Rules', 'macro_f1']) }}. Feature-request F1 of the deployed model is
{{ f2(dep['fr_f1']) }} {{ dep['fr_f1_ci'] }}; with only {{ gold['by_split'].loc['feature_request', 'test'] }}
feature requests in the test split, class-level CIs are wide.

Three further points stand out. First, the rules overfit: they scored {{ f3(dev.loc['rules', 'macro_f1']) }} on the dev
items they were written against but {{ f3(det.loc['Rules', 'macro_f1']) }} on test. Second, the LLM settings are
statistically indistinguishable from each other: the second LLM family, few-shot examples and prompt v2 all land
within each other's confidence intervals, although prompt v2 is consistently ahead of v1
({{ f3(det.loc['LLM zero-shot (prompt v1)', 'macro_f1']) }} → {{ f3(teacher['macro_f1']) }}, κ
{{ f2(det.loc['LLM zero-shot (prompt v1)', 'kappa']) }} → {{ f2(teacher['kappa']) }}). Third, the LLMs' κ with the
annotator ({{ f2(teacher['kappa']) }}) is below the annotator's own intra-annotator κ ({{ f2(gold['kappa']) }}), so
there is still room between the models and the human ceiling.

![Cost vs accuracy](figures/report_cost_accuracy.png)

Per-class F1:

{{ md(per_class.reset_index().rename(columns={'': 'class'})) }}

Praise is the easiest class for the LLMs and the silver-trained models. *feature_request* is the hardest or
second-hardest class for each of them, with *other* and *bug_report* close behind: these classes are rare and
border on several others.

### 6.2 Distillation (RQ2)

Trained on {{ gold['by_split'].loc['total', 'dev'] }} gold labels, SetFit reaches {{ f3(det.loc['SetFit (gold-dev)', 'macro_f1']) }}
and TF-IDF+LR {{ f3(det.loc['TF-IDF+LR (gold-dev)', 'macro_f1']) }}; trained on silver labels they reach
{{ f3(det.loc['SetFit (silver)', 'macro_f1']) }} and {{ f3(det.loc['TF-IDF+LR (silver)', 'macro_f1']) }}.

![Distillation](figures/report_distillation.png)

Silver labels close the gap to the teacher. With {{ gold['by_split'].loc['total', 'dev'] }} examples, a bag-of-words
model has too few examples per class to learn anything useful; SetFit's pre-trained encoder already does well,
but it still gains {{ f2(det.loc['SetFit (silver)', 'macro_f1'] - det.loc['SetFit (gold-dev)', 'macro_f1']) }} macro-F1 from
the silver data. The silver labels are noisy (the teacher agrees with the annotator on only
{{ pct(teacher['accuracy']) }} of test items), yet the student is not worse than the teacher. The student does not
exceed it either: it inherits the teacher's systematic mistakes (§6.3), which caps it at the teacher's level.

In practice this means a one-off LLM labelling run buys a model that classifies the whole corpus, and any new
feedback, offline, at {{ f3(dep['s_per_item']) }} s per item and with no per-item cost or quota.
The LLM is then needed only for extraction, on the much smaller set of predicted requests. Since the student
tracks its teacher, the way to improve it is to improve the teacher, as prompt v2 did, and to relabel.

### 6.3 Error analysis

The deployed model makes {{ errors['deployed'] }} errors on the test split and its teacher {{ errors['teacher'] }}.
Both are wrong on {{ errors['both_wrong'] }} items, and on {{ errors['same_label'] }} of them they predict the same wrong label,
which points to ambiguous items or annotator conventions. Most frequent confusions of the deployed model:

{{ md(errors['top'].head(5), pct_cols=('share_of_errors',)) }}

![Confusion matrix, deployed model](figures/detect_confusion_setfit_silver.png)

The examples in `reports/tables/detect_errors.md` show that most errors sit on the boundaries discussed in §4.

- **Complaint → feature request.** *"stop stopping my music to ask me to rate you!"* is an annoyance written as an
  imperative; by the letter of convention (1) it is arguably a request, and the teacher says so too. A long review
  explaining that Premium is forced on users and that YouTube lets them play whichever song they want contains an
  embedded wish (choose songs, go back and forward) inside a complaint. Some of these gold labels are debatable.
- **Complaint → bug report.** Downloaded playlists that stopped working on a camping trip describe a malfunction,
  but the reviewer's point is that Premium is unreliable offline. In *"AI music is destroying this app … AI is
  crashing and ruining it all"*, "crashing" is figurative, and the model reads it literally.
- **Question → other.** Reddit replies such as *"It has the same layouts as the Mix feature … Did you click that by
  any chance?"* are mostly information with a question attached; whether the question makes it a question is a
  fine call.
- **Bug report → question.** *"Spotify app auto closes when I launch it … Are you fixing this anytime soon???"* is a
  bug report with a rhetorical question, which the model takes literally.

These are the boundaries the guidelines and the prompt conventions try to pin down (bug vs complaint is also the
one on which the annotator changed labels between pilot passes), and most of these errors are shared with the
teacher. More silver data would not fix them; a second annotator and sharper guidelines would.

### 6.4 Extraction (RQ3a)

On the gold feature requests, the judge grades the LLM extraction correct for
**{{ pct(ext.loc[('gold all', 'llm'), 'judge_correct']) }}** of items and the rules for
{{ pct(ext.loc[('gold all', 'rules'), 'judge_correct']) }}; on {{ int(ext.loc[('corpus sample', 'llm'), 'items']) }} corpus
items the classifier labels as requests: {{ pct(ext.loc[('corpus sample', 'llm'), 'judge_correct']) }} vs
{{ pct(ext.loc[('corpus sample', 'rules'), 'judge_correct']) }}. Phrase-matching F1 against the annotator
(cosine ≥ 0.6): LLM {{ f3(ext.loc[('gold all', 'llm'), 'f1']) }} {{ ext.loc[('gold all', 'llm'), 'f1_ci'] }},
rules {{ f3(ext.loc[('gold all', 'rules'), 'f1']) }} {{ ext.loc[('gold all', 'rules'), 'f1_ci'] }}.

{{ md(ext.reset_index(), ['setting', 'method', 'items', 'precision', 'recall', 'f1', 'f1_ci', 'judge_correct', 'judge_partial', 'judge_wrong']) }}

Phrase-matching F1 is low for both methods because the annotator's phrases are terse and personal, while the LLM
writes product-style phrases ("fewer ads"); the judge grades are the better measure. The rules fail mainly on
implicit requests ("why can't I…", "Apple Music has…"), where no request pattern matches. The LLM also acts as a
precision filter: for classifier false positives it returns no feature, which the judge grades as correct, and
this contributes to its higher grade on the corpus sample than on gold.

### 6.5 Clustering (RQ3b)

{{ md(cmp, ['method', 'n_clusters', 'outlier_rate', 'outlier_mentions', 'silhouette', 'largest_share']) }}

Purity on sampled clusters (share of members that belong):

{{ md(pur, ['method', 'rater', 'clusters', 'members', 'purity', 'ci', 'fully_pure']) }}

On the {{ jh['members'] }} phrases checked by hand, the LLM judge agrees with the human on
{{ pct(jh['agreement']) }} (Cohen's κ **{{ f2(jh['kappa']) }}**) but is more lenient
({{ pct(jh['judge_intruders']) }} vs {{ pct(jh['human_intruders']) }} intruders), so LLM-judged purity
({{ f3(pur.loc['A LLM judge, human subset', 'purity']) }}) overstates human purity
({{ f3(pur.loc['A human', 'purity']) }}) on the same themes. Theme names fit their phrases for
{{ pct(jh['name_yes']) }} of themes and partly for {{ pct(jh['name_partly']) }}.

![Cluster purity](figures/report_purity.png)

BERTopic scores a higher purity ({{ f3(pur.loc['B LLM judge', 'purity']) }} vs
{{ f3(pur.loc['A LLM judge', 'purity']) }}), but the comparison flatters it. It clusters whole posts, so its
{{ cmp.loc['B', 'n_clusters'] }} topics are broad subjects (ads, shuffle, podcasts), and a review is easily judged to
"belong" to a broad subject even when its actual request differs. It also leaves
{{ pct(cmp.loc['B', 'outlier_rate']) }} of items out as outliers, against {{ pct(cmp.loc['A', 'outlier_mentions']) }}
of mentions for the phrase clusters, and it can put a review that makes two requests into only one topic. The
phrase clusters are narrow and specific, which makes intruders easier to spot and creates some near-duplicate
themes (wording splits that the conservative merge step did not join), but it is the specificity a ranking of
requests needs. For an overview of what people talk about, BERTopic is adequate; for a list of what people ask
for, phrase clustering is the better choice.

## 7. Insights (RQ4)

### 7.1 What users ask for

{{ md(top.head(10), ['rank', 'name', 'items', 'low_rating_share', 'growth', 'priority', 'trend'], pct_cols=('low_rating_share',)) }}

![Top themes](figures/top_themes.png)

### 7.2 Trends

Comparing the last {{ trends['weeks'] }} weeks ({{ n(trends['recent_items']) }} items) with the
{{ trends['weeks'] }} before ({{ n(trends['prev_items']) }}): **{{ trends['emerging'] }}** themes emerge significantly
and **{{ trends['declining'] }}** decline:

{{ md(trends['declining_table']) }}

Both declines are the aftermath of a spike rather than solved problems. The larger one,
"Compact Home Screen Widgets"
({{ trends['declining_table'].set_index('name').loc['Compact Home Screen Widgets', 'prev_items'] }} → {{ trends['declining_table'].set_index('name').loc['Compact Home Screen Widgets', 'recent_items'] }} items),
followed an app update in June 2026 that replaced the small Android home-screen widget with a larger one: the
requests to bring the old widget back are concentrated in the days after that release and in that app version,
then fade, but they do not stop. Requests for the classic app logo likewise peak in May 2026 and then fade.
No theme grows significantly: the request mix of the last quarter is much like that of the quarter before.

### 7.3 Platform and rating differences

Most Reddit-leaning themes: {{ names(ins['reddit_leaning']) }}. Most Play-leaning: {{ names(ins['play_leaning']) }}.
Themes in the lowest-rated Play reviews: {{ names(ins['angriest']) }}.

The two platforms ask for different things. Reddit users, an enthusiast community, ask for customisation and
control of their library. Play reviewers ask about the limits of the free tier: playback order,
downloads, rewinding and payment options. The lowest-rated reviews are dominated by requests to switch something
*off* (AI-generated music, AI features, recommendations, podcasts) rather than to add something.

### 7.4 Competitors

Most cited: {{ ', '.join(f"{r.competitor} ({n(r.items)})" for r in ins['competitors'].head(5).itertuples()) }}.

YouTube Music is cited far more often than any other service, mostly alongside requests about ads, price and
skips, i.e. as the free alternative. Apple Music is cited mostly alongside requests for lossless, high-quality
audio.

### 7.5 Requests for features that already exist

Of {{ grid['themes'] }} themes, {{ ins['status']['exists'] }} ask for something that already exists for the users
asking, usually for everyone (discoverability), {{ ins['status']['premium_only'] }} for Premium-only features on the free tier (paywall),
{{ ins['status']['partly'] }} for more than a limited version offers, and {{ ins['status']['new'] }} are new.

Largest "already exists" themes:

{{ md(ins['exists'].head(6)) }}

### 7.6 What Spotify could do with this

- **Free-tier limits are the main request.** The top of the ranking ({{ names(top, 3) }}) is about the
  business model, not missing functionality. These are pricing and packaging decisions; the volume and the low
  ratings attached to them suggest room for options between the free tier and full Premium.
- **Give users an off switch.** Opt-outs from AI music, AI features, recommendations and podcasts sit in the
  angriest reviews. Settings toggles and labelling of AI-generated tracks are cheap compared with the
  dissatisfaction they would address.
- **Fix discoverability before building.** {{ ins['status']['exists'] }} themes ask for features that already
  exist, the largest being "{{ ins['exists']['name'].iloc[0] }}" (*{{ ins['exists']['feature'].iloc[0] }}*).
  Better in-app discovery, or a reply to the review pointing to the feature, would answer these users.
- **Watch reactions to releases.** "Bring back …" themes, such as the widget, spike after an update. Tracking them
  per app version gives early warning that a change is unpopular.
- **Listen on more than one channel.** Play and Reddit produce different priority lists; deciding from one of them
  alone would miss either the free-tier majority or the power users.

### 7.7 The dashboard

The dashboard (`make dashboard`) presents the results in six pages. It only reads files written by the pipeline,
so it runs offline. A filter row for sources and months is shared by all pages.

- **Overview**: headline numbers, the top themes and the full ranking table. Sliders change the four priority
  weights and show how each theme's rank moves.
- **Request Explorer**: one theme in depth: common phrasings, monthly share, sources, star ratings compared with
  all Play reviews, competitors and example quotes, sortable by confidence or engagement.
- **Trends**: growth against size for every theme, the biggest movers and small multiples of chosen themes.
- **Insights**: platform lean, rating profile, competitors and the "already exists?" status of each theme.
- **Model Lab**: the evaluation of every stage (detection with CIs and confusion matrices, distillation, errors,
  extraction and clustering, including the human purity check).
- **Live Analyser**: classifies new text with the local model, extracts its features (rules, or the LLM as an
  opt-in) and finds the nearest existing themes, flagging possibly new requests.

![Dashboard: Overview](figures/dashboard_overview.png)

![Dashboard: Request Explorer](figures/dashboard_explorer.png)

![Dashboard: Trends](figures/dashboard_trends.png)

![Dashboard: Insights](figures/dashboard_insights.png)

![Dashboard: Model Lab](figures/dashboard_model_lab.png)

![Dashboard: Live Analyser](figures/dashboard_live.png)

## 8. Limitations and Ethics

- Single annotator (intra-annotator κ {{ f2(gold['kappa']) }}); some gold labels are debatable.
  Without a second annotator we cannot separate model error from label noise on the hard boundaries, and the
  intra-annotator κ comes from a small pilot of {{ gold['pilot_items'] }} items.
- Only {{ gold['by_split'].loc['feature_request', 'test'] }} feature requests in gold test: wide CIs for class-level metrics.
  The deployed model's feature-request F1 has the interval {{ dep['fr_f1_ci'] }}, so most differences between systems
  on that class are not significant; this is why macro-F1 is the primary metric.
- Extraction quality and most purity numbers are LLM judgments; the human check (κ {{ f2(jh['kappa']) }}) shows the judge is lenient.
  The judge comes from a different model family than the extractor to avoid self-preference, but the human check
  shows it still flags fewer intruders than a person, so LLM-judged purity should be read as an upper bound.
- Clustering over-splits some requests by wording; the merge step is conservative ({{ grid['confirmed'] }} of {{ grid['proposed_merges'] }} proposals kept).
  A few near-duplicate themes remain (for example, two on-demand playback themes), which splits their volume and
  lowers their rank.
- "Already exists" statuses depend on a hand-written catalogue and are sensitive to the prompt.
  The matching LLM is not stable between runs: running it twice with the same catalogue and prompt (once after
  the catalogue was corrected, once again on another machine) moved several themes between statuses, mostly
  small ones and mostly between *exists*, *partly* and *premium_only*. The per-theme statuses are therefore
  indicative; the overall split and the largest themes are more reliable than any single status.
- {{ corpus_note }}
  Trends leave 5★ Play reviews out for this reason. The rating statistics (≤2★ share, mean rating) use all Play
  reviews, so their absolute levels over-represent low ratings; comparisons between themes, and between a theme
  and all Play reviews, are less affected because the same bias applies across them.
- LLM non-determinism (temperature 0, cached responses), sampling bias of reviewers and Reddit users, platform terms of service.
  Even at temperature 0, re-running an LLM stage without the cache can give slightly different results; the cached
  responses make the reported numbers reproducible. Reviewers and Reddit users are self-selected and not
  representative of all Spotify users: angry users and enthusiasts are over-represented, and the data are English
  only and Android only. Data collection respected the ethical constraints in §3.

## 9. Conclusion and Future Work

**RQ1.** Feature requests can be detected in noisy social text with moderate accuracy. The best systems reach
macro-F1 of about {{ f2(teacher['macro_f1']) }} over six classes and feature-request F1 of about
{{ f2(dep['fr_f1']) }}. Rules are cheap but overfit their development data; LLMs from two families perform alike,
and prompt conventions helped more than few-shot examples, although neither difference is significant. The main
errors sit on the boundaries that were hardest to define in the guidelines.

**RQ2.** Yes. A SetFit model trained on {{ n(det.loc['SetFit (silver)', 'train'].split()[0]) }} LLM labels matches its
teacher ({{ f3(dep['macro_f1']) }} vs {{ f3(teacher['macro_f1']) }}), beats the same model trained on
{{ gold['by_split'].loc['total', 'dev'] }} hand labels, and runs locally at no per-item cost. It inherits the teacher's
mistakes, so it is unlikely to exceed the teacher.

**RQ3.** LLM extraction works well ({{ pct(ext.loc[('gold all', 'llm'), 'judge_correct']) }} correct on gold,
{{ pct(ext.loc[('corpus sample', 'llm'), 'judge_correct']) }} on the corpus sample) and clearly beats dependency rules,
mostly because it handles implicit requests. Phrase clustering produces specific, usable themes with a human
purity of {{ f2(pur.loc['A human', 'purity']) }}, but it splits some requests by wording, and LLM-judged purity
overstates quality.

**RQ4.** Spotify users mostly ask to loosen the free tier: {{ names(top, 3) }} lead the ranking. Requests differ
strongly by platform: Reddit asks for customisation, Play for free-tier limits. The angriest reviews ask to turn
features off. {{ ins['status']['exists'] }} themes ask for features that already exist and
{{ ins['status']['premium_only'] }} for Premium-only ones, pointing to discoverability and pricing rather than
missing functionality.

**Future work.** A second annotator to measure inter-annotator agreement and settle the hard boundaries; a larger
gold set with more feature requests; App Store data and multilingual reviews; active learning that sends the
student's least confident items to the LLM or a human; a domain-adapted encoder for clustering, which might reduce
wording splits; and running the pipeline continuously so that requests can be tracked per app release.

## References

Campello, R. J. G. B., Moulavi, D., and Sander, J. (2013). Density-based clustering based on hierarchical density
estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, LNCS 7819, pp. 160–172. Springer.

Chen, N., Lin, J., Hoi, S. C. H., Xiao, X., and Zhang, B. (2014). AR-Miner: Mining informative reviews for
developers from mobile app marketplace. In *Proceedings of the 36th International Conference on Software
Engineering (ICSE 2014)*, pp. 767–778. ACM.

Gilardi, F., Alizadeh, M., and Kubli, M. (2023). ChatGPT outperforms crowd workers for text-annotation tasks.
*Proceedings of the National Academy of Sciences*, 120(30), e2305016120.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint
arXiv:2203.05794*.

Guzman, E., and Maalej, W. (2014). How do users like this feature? A fine grained sentiment analysis of app
reviews. In *2014 IEEE 22nd International Requirements Engineering Conference (RE)*, pp. 153–162. IEEE.

Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the knowledge in a neural network. *arXiv preprint
arXiv:1503.02531*.

Maalej, W., and Nabil, H. (2015). Bug report, feature request, or simply praise? On automatically classifying app
reviews. In *2015 IEEE 23rd International Requirements Engineering Conference (RE)*, pp. 116–125. IEEE.

McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension
reduction. *arXiv preprint arXiv:1802.03426*.

Nguyen, D. Q., Vu, T., and Nguyen, A. T. (2020). BERTweet: A pre-trained language model for English Tweets. In
*Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*,
pp. 9–14. ACL.

Panichella, S., Di Sorbo, A., Guzman, E., Visaggio, C. A., Canfora, G., and Gall, H. C. (2015). How can I improve
my app? Classifying user reviews for software maintenance and evolution. In *2015 IEEE International Conference
on Software Maintenance and Evolution (ICSME)*, pp. 281–290. IEEE.

Reimers, N., and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In
*Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP)*,
pp. 3982–3992. ACL.

Tunstall, L., Reimers, N., Jo, U. E. S., Bates, L., Korat, D., Wasserblat, M., and Pereg, O. (2022). Efficient
few-shot learning without prompts. *arXiv preprint arXiv:2209.11055*.

Wang, S., Liu, Y., Xu, Y., Zhu, C., and Zeng, M. (2021). Want to reduce labeling cost? GPT-3 can help. In
*Findings of the Association for Computational Linguistics: EMNLP 2021*, pp. 4195–4205. ACL.
