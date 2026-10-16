# Annotation Guidelines

Gold-standard labelling for the feature-request detection experiments (Phase 2).

**The task:** read one piece of Spotify user feedback (a Play Store review, a Reddit post, or a Reddit comment) and assign **exactly one** label describing its main communicative intent. If the item is a feature request, also write down *what* is being requested.

Label from **the text alone**. Do not guess at context you cannot see (for example, the post a comment replies to). The models will see only the text, so the gold labels must be judged from the same information.

---

## 1. Labels

| Key | Label | Definition | Typical signals |
|---|---|---|---|
| `1` | **feature_request** | Asks for new functionality, or for a change to how existing functionality, content, pricing or ads work. The request can be explicit or implicit. | "please add", "I wish", "should let us", "bring back", "why can't I…", "Apple Music has X…" |
| `2` | **bug_report** | Something is broken or not working as intended: crashes, errors, playback failures, sync problems, billing/account malfunctions. | "keeps crashing", "won't load", "since the update X doesn't work", "charged twice" |
| `3` | **complaint** | Negative opinion or frustration, with no concrete request and no concrete malfunction. | "too many ads", "getting worse every update", "overpriced garbage" |
| `4` | **praise** | Positive opinion with no request. | "best music app", "love the Discover Weekly" |
| `5` | **question** | Seeks information or help: how to do something, whether something exists, why something happens. | "how do I…", "is there a way to…", "does anyone know…" |
| `6` | **other** | None of the above: news or info sharing, off-topic, spam, unintelligible, jokes. | "Spotify Wrapped drops Dec 3", "first!!", random text |

Also:
- **`u` = unsure.** Toggle it whenever you hesitate between two labels. Still pick your best guess. Unsure items are analysed separately and are not errors.
- **Requested feature** (feature_request only): a short noun phrase for each thing requested, separated by `;`. For example `sleep timer for podcasts` or `hide songs from playlists; offline lyrics`. Keep it short and normalised. Leave it blank if you can't tell what exactly is requested.

---

## 2. Decision rules

Work through these **in order**. The first rule that applies wins.

1. **Does the text ask for something to be added, changed, restored or removed, explicitly or implicitly?** → `feature_request`
   - This counts even when the request sits inside a complaint, a bug report or praise.
2. **Does it describe a concrete malfunction?** → `bug_report`
3. **Does it seek information or help?** → `question`
4. **Is it a negative opinion?** → `complaint`
5. **Is it a positive opinion?** → `praise`
6. Otherwise → `other`

This gives the priority **feature_request > bug_report > question > complaint > praise > other** for mixed items. Feature requests come first because they are the target of the study.

---

## 3. Hard cases

### Feature request vs. bug report
| Text | Label | Why |
|---|---|---|
| "Shuffle keeps playing the same 20 songs, please fix" | bug_report | Describes a malfunction. "Please fix" asks for a repair, not a new feature. |
| "Please make shuffle actually random, or let me choose a true-random mode" | feature_request | Asks for a behaviour change or a new option. |
| "Bring back the old Now Playing screen" | feature_request | Asks to restore a deliberate design change. |
| "Lyrics disappeared after the update" | bug_report | Reads as a malfunction, unless the text says it was removed on purpose. |
| "They removed the lyrics for free users, bring them back" | feature_request | A deliberate change the user wants reversed. |

### Feature request vs. complaint
The test: **is there an actionable ask?**

| Text | Label |
|---|---|
| "Way too many ads now" | complaint |
| "Way too many ads, please reduce them or let us skip" | feature_request (feature: `fewer ads; skippable ads`) |
| "Premium is overpriced" | complaint |
| "Add a cheaper student plan in my country" | feature_request |

Requests about **ads, pricing and plans count as feature requests**. They are changes users ask for, and the analysis can separate business requests from functional ones later.

### Feature request vs. question
| Text | Label | Why |
|---|---|---|
| "Is there a way to hide a song from a playlist?" | question | Genuinely asking whether it exists. |
| "Why is there STILL no way to hide songs??" | feature_request | Rhetorical. The user knows it doesn't exist and wants it. |
| "Will Spotify ever add a sleep timer on desktop?" | feature_request | Implies a desire for the feature. |
| "How do I turn off autoplay?" | question | Asks how to do something. |

### Implicit requests (all feature_request)
- "Apple Music lets you see who wrote each song, just saying" → feature: `songwriter credits`
- "Would kill for a dark mode on the web player 💀" → feature: `dark mode for web player`
- "5 stars if you add karaoke" → feature: `karaoke mode`

### Mixed and other cases
- **Praise + request:** "Love the app! Please add a queue history" → `feature_request`
- **Complaint + malfunction:** "Garbage app, crashes every time I open it" → `bug_report`
- **Sarcasm:** label the intended meaning. "Oh great, another redesign nobody asked for" → `complaint`
- **News or info posts:** "Spotify is testing lossless in 3 markets" → `other`. But "Spotify is testing lossless, finally, roll it out everywhere please" → `feature_request`.
- **Reddit comment agreeing with an unseen request:** "+1, I want this too" → judge the text alone. Nothing concrete is requested, so `other` (mark unsure if you like).
- **Account and billing problems** ("charged after cancelling", "can't log in") → `bug_report`
- **Content requests** ("add more Bollywood songs", "please get Taylor Swift's old albums back") → `feature_request`, feature `more <genre/artist> content`

---

## 4. Practical tips

- Aim for **≈10 seconds per item**. Read once, decide, move on. Use `u` instead of agonising.
- Long Reddit posts: the title often states the intent. Skim the body for an explicit ask.
- Take a break every ~100 items. Label quality drops when you're tired.
- Your labels are saved after every item, so you can stop and resume at any time.

---

## 5. Process notes (for the report)

- Single annotator (the author). Labels are assigned **without** seeing any model prediction, to avoid anchoring.
- Guidelines were fixed before labelling. Any later clarification is recorded in the changelog below and applied to all items.
- The dev split (100 items) is labelled first and used for prompt/rule development. The test split (300 items) is used only for the final evaluation.

### Changelog
- v1.0 (2026-10-16): initial version.
