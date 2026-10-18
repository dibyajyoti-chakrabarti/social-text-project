# Demo script (5–7 minutes)

**Setup before the demo:** `make dashboard`, open http://localhost:8501, and run one analysis
on the Live Analyser once so the models are loaded (the first run takes a few seconds). Wi-Fi is not needed;
leave the LLM toggle **off**. Reset filters to all sources and all months.

---

## 0:00 · The problem (30 s) · *Overview page*

> "Spotify gets thousands of reviews and Reddit posts every week. Hidden in them are feature
> requests, but they are a small slice: in our hand-labelled sample only
> {{ pct(gold['by_split'].loc['feature_request', 'share'], 1) }} of items were feature requests. We built a pipeline
> that finds them, extracts what is being asked for, groups the asks into themes and ranks them."

Point at the KPI row: {{ n(corpus['english']) }} items analysed over a year from Google Play and r/truespotify,
{{ n(grid['mentions']) }} extracted feature mentions, {{ grid['themes'] }} themes.

## 0:30 · What users want (60 s) · *Overview*

- The top-ranked themes: {{ names(top, 5) }}.
- "{{ top['name'].iloc[0] }}" alone has {{ n(top['items'].iloc[0]) }} items.
- **Interaction:** open *Priority weights*, drag *Growth* to 1.0 → the ranking re-orders and the Δ rank column shows
  which themes move. Reset.
- **Interaction:** switch *Sources* to Reddit only → the request mix changes: Reddit asks for customisation,
  Play reviewers for free-tier limits. Switch back.

## 1:30 · One theme in depth (60 s) · *Request Explorer*

- Pick "{{ top['name'].iloc[1] }}" (rank 2): monthly share line, who asks, star ratings compared with all Play reviews.
- Scroll to *What people say*: real quotes with the phrase our extractor pulled out.
- Mention the competitors line: users name YouTube Music most often
  ({{ n(ins['competitors']['items'].iloc[0]) }} items overall).

## 2:30 · What is changing (45 s) · *Trends*

- "No theme is significantly emerging in the last quarter: {{ trends['emerging'] }} emerging, {{ trends['declining'] }} declining."
- Point at "Compact Home Screen Widgets" ({{ trends['declining_table'].set_index('name').loc['Compact Home Screen Widgets', 'prev_items'] }} → {{ trends['declining_table'].set_index('name').loc['Compact Home Screen Widgets', 'recent_items'] }} items):
  "This one spiked in June, when an app update replaced the small Android widget with a larger one. People
  asked for the old one back, then the requests faded, but they never stopped." Open it in the Request Explorer
  to show the June spike if time allows.

## 3:15 · Insights for the product team (45 s) · *Insights → Already exists?*

- {{ ins['status']['exists'] }} themes ask for features Spotify **already has**, e.g.
  "{{ ins['exists']['name'].iloc[0] }}" → *{{ ins['exists']['feature'].iloc[0] }}*: a discoverability problem, not a missing feature.
- {{ ins['status']['premium_only'] }} themes ask for Premium features on the free tier: a paywall signal.
- Optional: *Ratings* tab, where requests to switch AI features off sit in the angriest reviews.

## 4:00 · Does it work? (75 s) · *Model Lab*

- *Detection* tab: "Gemini labels {{ n(det.loc['SetFit (silver)', 'train'].split()[0]) }} reviews; a small SetFit model
  trained on those labels scores macro-F1 {{ f2(dep['macro_f1']) }} on our held-out gold test, the same as Gemini itself
  ({{ f2(teacher['macro_f1']) }}), and runs locally, about {{ int(round(teacher['s_per_item'] / dep['s_per_item'])) }}× faster per item."
- Point at the *cost vs accuracy* chart and the *Distillation* tab: trained on only 100 hand labels it reaches
  {{ f2(det.loc['SetFit (gold-dev)', 'macro_f1']) }}.
- *Clustering* tab: "We also checked the themes by hand: human purity {{ f2(pur.loc['A human', 'purity']) }};
  the LLM judge agrees with us at κ {{ f2(jh['kappa']) }} but is more lenient, which is why we report both."

## 5:15 · Live (60 s) · *Live Analyser*

- Click the *Feature request* example → **Analyse**: class probabilities from the local model, extracted features,
  and the nearest themes with cosine similarity.
- Type your own sentence, e.g. *"please let me hide podcasts from my home screen"*.
- With the toggle off, the rules extract a stray *"app"* from "Love the app" next to the real request. Say so
  rather than hide it: it is exactly why the rules are graded correct for only
  {{ pct(ext.loc[('gold all', 'rules'), 'judge_correct']) }} of gold requests.
- "Everything you've seen ran offline. The LLM toggle switches extraction to Gemini, which is right
  {{ pct(ext.loc[('gold all', 'llm'), 'judge_correct']) }} of the time on our gold set versus
  {{ pct(ext.loc[('gold all', 'rules'), 'judge_correct']) }} for the rules, at the cost of an API call."

## 6:15 · Wrap-up (30 s)

> "One pipeline, from raw reviews to a ranked, explorable list of what users want, with every
> number reproducible from scripts. Limitations: a single annotator, few feature requests in the
> test set, and LLM-judged extraction. Questions?"

---

**If something goes wrong:** the dashboard only reads files, so restarting `make dashboard` fixes most
issues. If the Live Analyser is slow, it is still loading the models; wait for the spinner.
