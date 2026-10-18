# Cluster purity

Random sample of 30 clusters per method × 10 random members. `purity`: share of members that belong (95% bootstrap CI over clusters); `fully_pure`: clusters with no intruder. Method A members are feature phrases, method B members are full posts/reviews.

| method             | rater                   |   clusters |   members |   purity | ci           |   fully_pure |
|:-------------------|:------------------------|-----------:|----------:|---------:|:-------------|-------------:|
| A: phrase clusters | LLM judge               |         30 |       300 |    0.797 | [0.72, 0.86] |         0.3  |
| B: BERTopic        | LLM judge               |         30 |       300 |    0.867 | [0.81, 0.92] |         0.4  |
| A: phrase clusters | human                   |         20 |       200 |    0.685 | [0.61, 0.76] |         0.05 |
| A: phrase clusters | LLM judge, human subset |         20 |       200 |    0.77  | [0.68, 0.85] |         0.15 |


## Judge vs human (method A)

200 members: agreement 0.83, Cohen's κ 0.59; the judge calls 23% intruders, the human 32%.

Theme name fits its phrases (human): yes 60%, partly 40%


## Intruders flagged by the judge (method A, first 15)

- **Library Folder Customization**: share folders
- **Pin More Playlists**: pin songs to top of playlist
- **Pin More Playlists**: pin more sidebar items
- **Home Screen Customization**: auto refresh home page
- **Album Art and Layout Customization**: android auto show album name
- **Album Art and Layout Customization**: filter albums by custom tag
- **Sequential Playlist Playback on Free Tier**: free tier playlist editing
- **Sequential Playlist Playback on Free Tier**: play on repeat on free tier
- **Sequential Playlist Playback on Free Tier**: play saved playlists on free tier
- **Sequential Playlist Playback on Free Tier**: remove play limits on free tier
- **Sequential Playlist Playback on Free Tier**: choose songs in playlists on free tier
- **Sequential Playlist Playback on Free Tier**: public playlists on free tier
- **Sequential Playlist Playback on Free Tier**: choose music in playlists on free tier
- **Sequential Playlist Playback on Free Tier**: play repeats on free tier
- **AI Music Labeling**: ai music assistant
