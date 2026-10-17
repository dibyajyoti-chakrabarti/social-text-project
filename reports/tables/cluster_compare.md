# Phrase clusters (A) vs BERTopic on item texts (B)

Silhouettes are computed in each method's own embedding space (phrases vs texts), so they are not directly comparable; purity (cluster_eval.md) is the shared measure.

| method                      | unit            |   n_clusters |   outlier_rate |   outlier_mentions |   silhouette |   largest_share |
|:----------------------------|:----------------|-------------:|---------------:|-------------------:|-------------:|----------------:|
| A: phrases + UMAP + HDBSCAN | distinct phrase |          144 |          0.065 |              0.036 |        0.13  |           0.147 |
| B: BERTopic on texts        | item            |           52 |          0.342 |              0.342 |        0.066 |           0.086 |

## Top themes (A)

|   theme_id | name                            |   items |   mentions |   n_phrases |
|-----------:|:--------------------------------|--------:|-----------:|------------:|
|          0 | Fewer and Shorter Ads           |    2709 |       2785 |         127 |
|          1 | Unlimited Song Skips            |    1082 |       1091 |          98 |
|          2 | Cheaper Premium Plans           |     792 |        812 |          52 |
|          3 | On-Demand Song Selection        |     624 |        625 |         121 |
|          4 | True Random Shuffle             |     492 |        503 |         136 |
|          5 | Synced Lyrics Display           |     425 |        449 |         252 |
|          6 | Compact Home Screen Widgets     |     403 |        426 |          85 |
|          7 | Disable Forced Shuffle          |     396 |        399 |          76 |
|          8 | Expanded Free Tier Features     |     358 |        366 |          73 |
|          9 | Ad Removal and Blocking         |     338 |        340 |          77 |
|         10 | On-Demand Playback              |     314 |        315 |          74 |
|         11 | Free Tier Enhancements          |     284 |        284 |          91 |
|         12 | AI Music Filter                 |     271 |        278 |          85 |
|         13 | Disable AI Features             |     254 |        255 |          97 |
|         14 | Free Tier On-Demand Playback    |     252 |        252 |          42 |
|         15 | Disable Video Playback          |     239 |        246 |         142 |
|         16 | Playlist Management and Variety |     223 |        236 |         210 |
|         17 | Offline Downloads for Free Tier |     229 |        229 |          53 |
|         18 | Song Repeat Feature             |     211 |        213 |          96 |
|         19 | Hide Podcasts                   |     170 |        179 |          91 |
|         20 | Hi-Fi Audio Quality             |     163 |        171 |          85 |
|         21 | Enhanced Liked Songs Management |     152 |        165 |         142 |
|         22 | Unlimited Audiobook Listening   |     154 |        163 |         102 |
|         23 | Free Queue Management           |     154 |        159 |         127 |
|         24 | Flexible Subscription Pricing   |     152 |        158 |          79 |
|         25 | Removal of Playback Limits      |     155 |        157 |          82 |
|         26 | Ad-Free Premium Experience      |     143 |        144 |          88 |
|         27 | Song Rewind Controls            |     143 |        143 |          50 |
|         28 | Free Premium Access             |     142 |        142 |          66 |
|         29 | Ad-Free Podcasts                |     136 |        141 |          47 |

## Top topics (B)

|   topic |   items | words                                               |
|--------:|--------:|:----------------------------------------------------|
|       0 |     809 | spotify, premium, free, app, listen, pay            |
|       1 |     784 | app, ads, premium, song, songs, listen              |
|       2 |     724 | ai, dj, slop, artists, generated, music             |
|       3 |     618 | ads, spotify, ad, premium, free, music              |
|       4 |     494 | shuffle, smart, songs, playlist, order, random      |
|       5 |     454 | lyrics, language, english, song, karaoke, songs     |
|       6 |     407 | playlist, liked, playlists, add, button, sort       |
|       7 |     397 | ads, reduce, remove, ad, stop, good                 |
|       8 |     314 | podcast, podcasts, episodes, episode, block, feed   |
|       9 |     274 | widget, old, new, 4x1, screen, bring                |
|      10 |     265 | premium, song, play, choose, songs, want            |
|      11 |     224 | ads, ad, song, music, songs, stop                   |
|      12 |     193 | light, dark, mode, ui, new, bar                     |
|      13 |     190 | ads, app, good, reduce, ad, great                   |
|      14 |     178 | device, phone, jam, car, bluetooth, automatically   |
|      15 |     174 | playlists, playlist, spotify, songs, new, artists   |
|      16 |     172 | premium, price, wish, free, lower, cheaper          |
|      17 |     150 | video, videos, disable, turn, home, watch           |
|      18 |     138 | podcasts, podcast, ads, premium, paying, pay        |
|      19 |     124 | offline, wifi, internet, downloaded, download, play |
|      20 |     122 | app, premium, free, cheaper, price, pay             |
|      21 |     117 | books, hours, book, audiobooks, audiobook, audio    |
|      22 |     115 | ads, premium, ad, songs, song, free                 |
|      23 |     109 | download, app, good, music, wish, songs             |
|      24 |     103 | adds, reduce, add, good, remove, pls                |
|      25 |     101 | shuffle, songs, spotify, playlist, random, order    |
|      26 |     100 | subscription, payment, plan, plans, bring, duo      |
|      27 |      91 | ads, premium, ad, subscription, buy, paying         |
|      28 |      90 | stats, capsule, sound, monthly, listening, bring    |
|      30 |      86 | queue, add, song, playlist, adding, clear           |
