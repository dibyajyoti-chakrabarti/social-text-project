# Phrase clustering: HDBSCAN grid

8370 distinct phrases (19669 mentions), `sentence-transformers/all-MiniLM-L6-v2` embeddings, UMAP to 5 dims. Noise phrases with cosine ≥ 0.6 to a centroid are attached after clustering. `outlier_mentions`: share of mentions left without a theme; `silhouette`: cosine, original embeddings. Selection rule: highest silhouette with outlier_mentions ≤ 0.2 and 30–250 clusters → min_cluster_size=20, min_samples=5.

|   min_cluster_size |   min_samples |   n_clusters |   outlier_rate |   outlier_mentions |   silhouette |   largest_share |
|-------------------:|--------------:|-------------:|---------------:|-------------------:|-------------:|----------------:|
|                  5 |             2 |          605 |          0.028 |              0.014 |        0.144 |           0.13  |
|                  5 |             5 |          470 |          0.038 |              0.019 |        0.156 |           0.136 |
|                  8 |             2 |          392 |          0.036 |              0.018 |        0.145 |           0.13  |
|                  8 |             5 |          343 |          0.04  |              0.019 |        0.144 |           0.144 |
|                 12 |             2 |          267 |          0.042 |              0.02  |        0.135 |           0.141 |
|                 12 |             5 |          237 |          0.045 |              0.022 |        0.132 |           0.145 |
|                 20 |             2 |          162 |          0.056 |              0.032 |        0.121 |           0.143 |
|                 20 |             5 |          152 |          0.065 |              0.036 |        0.134 |           0.147 |
|                 30 |             2 |          110 |          0.088 |              0.051 |        0.123 |           0.151 |
|                 30 |             5 |          100 |          0.094 |              0.054 |        0.12  |           0.15  |

LLM consolidation: gemini-3.5-flash-lite proposed 17 groups of same-request clusters (25 merges); openai/gpt-oss-20b confirmed 8 of them as pairwise yes/no questions. Confirmed merges joined 15 clusters into 7, leaving 144 themes (silhouette 0.130).

Kept merges:

- [remove premium paywall; remove paywalls] + [free premium features; free premium subscription]
- [unlimited song skips; free song skipping] + [remove skip limit; unlimited skips]
- [audiobooks included in premium; free audiobooks] + [unlimited audiobook listening; more audiobook listening hours]
- [replay songs; free song replay] + [repeat song; repeat songs] + [bring back repeat button; repeat button]
- [free song scrubbing; bring back song scrubbing] + [seek within tracks on free tier; skip tracks]
- [bring back old widget; bring back small widget] + [small widget; better widgets]
- [select specific songs; choose specific songs] + [free song selection; on-demand song selection on free tier]

Rejected merges (largest cluster ← rejected member):

- [remove skip limit; unlimited skips] ← [clear queue button; free queue management]
- [true random shuffle; better shuffle algorithm] ← [turn off shuffle on free tier; disable shuffle on free tier]
- [true random shuffle; better shuffle algorithm] ← [better playlist shuffle; improve playlist shuffle]
- [true random shuffle; better shuffle algorithm] ← [disable smart shuffle; disable forced shuffle]
- [disable podcasts; disable podcast autoplay] ← [disable podcast captions; turn off podcast transcripts]
- [hide podcasts; remove podcasts] ← [remove audiobooks; hide audiobooks]
- [fewer ads; shorter ads] ← [remove podcast ads; fewer podcast ads]
- [fewer ads; shorter ads] ← [remove ads; skip ads]
- [block ai artists; block artists] ← [block podcasts; block specific podcasts]
- [higher audio quality; better audio quality] ← [lossless audio; lossless audio quality]
- [ad-free tier; remove ads on free tier] ← [ad-free premium experience; free downloads with ads]
- [remove ai dj; disable ai dj] ← [sort podcast episodes by newest; chronological podcast playback]
- [bring back rewind button; rewind on free tier] ← [bring back skip button; fast forward button]
- [offline downloads on free tier; free offline downloads] ← [offline playback; offline music download]
- [free song selection; on-demand song selection on free tier] ← [on demand song playback; on-demand song playback]
- [free song selection; on-demand song selection on free tier] ← [on demand playback on free tier; on-demand playback on free tier]
- [free song selection; on-demand song selection on free tier] ← [free tier song selection; select specific songs on free tier]
