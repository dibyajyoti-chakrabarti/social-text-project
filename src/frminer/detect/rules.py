"""Tier 1: transparent keyword/regex classifier.

Each class has a list of patterns. Classes are checked in the guideline priority order
(feature_request > bug_report > question > complaint > praise > other) and the first
class with a matching pattern wins. Patterns were written from the guidelines and tuned
on the dev split only.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Verbs a user asks Spotify to do (excluding "fix", which signals a bug per the guidelines).
_ASK = r"(add|bring|make|let|allow|give|remove|put|change|stop|reduce|implement|include|return|restore|enable|introduce|lower|update)"
_COMPETITORS = r"(apple music|youtube music|yt music|tidal|deezer|amazon music|soundcloud|pandora)"

PATTERNS: dict[str, list[str]] = {
    "feature_request": [
        rf"\b(please|pls|plz|kindly)\b[^.!?]{{0,40}}\b{_ASK}\b",
        r"\b(i|we) (really )?(wish|hope)\b",
        r"\bwould (be (nice|great|cool|awesome|better)|love|like)\b[^.!?]{0,40}\b(if|to have|option|feature|ability|to see|to be able)\b",
        rf"\b(should|needs? to|have to|must|could you|can you|you guys need to)\b[^.!?]{{0,15}}\b({_ASK}|have|be able)\b",
        r"\bbring (back|it back|them back)\b",
        r"\b(add|adding) (a |an |the |more |some |back |option|feature|ability|support)",
        r"\b(option|ability|feature|setting|toggle|button|way) to\b",
        r"\bwhy (can'?t|cannot|can not|is there no|isn'?t there|no|don'?t you|won'?t you|not)\b",
        r"\b(feature request|suggestion|i suggest|i request|request:)",
        r"(?<!n't )(?<!not )(?<!wont )\blet (us|me|users|people|free users)\b",
        r"\bgive (us|me|users) (the|an?|more|back|option)\b",
        r"\b\d ?stars? (if|when|once|after)\b",
        rf"\b{_COMPETITORS}\b[^.!?]{{0,30}}\b(has|have|lets|allows|does|offers)\b",
        r"\b(fewer|less) ads\b",
        r"\bwant (the|an?|my|our) [^.!?]{0,30}\bback\b",
        r"\b(reduce|lower|cheaper)\b[^.!?]{0,30}\b(price|prices|plan|ads)\b",
        r"\b(put|change|turn|switch) it back\b",
    ],
    # Help-seeking about a problem ("anyone else having this?") counts as a question,
    # which is how such Reddit troubleshooting posts were labelled in dev.
    "help_seeking": [
        r"\b(anyone else|any (tips|fix|fixes|ideas|solutions?|advice|help)|know (of )?(a|any) fix|how (do|can) i fix|is this (a|just)|happening to anyone)\b",
    ],
    "bug_report": [
        r"\b(crash|crashes|crashing|crashed|freez\w*|glitch\w*|bug|bugs|buggy|error)\b",
        r"\b(not|isn'?t|wasn'?t|aren'?t|stopped|stops) (working|loading|playing|opening|showing|syncing|responding)\b",
        r"\b(doesn'?t|does not|won'?t|wont|will not|can'?t|cannot|didn'?t|couldn'?t) (work|load|play|open|connect|sync|download|log|sign|login|start|show|let me (log|sign|play|listen))\b",
        r"\bkeeps? (stopping|pausing|skipping|crashing|closing|logging|buffering|freezing|repeating)\b",
        r"\b(black|blank|white) screen\b",
        r"\b(charged|refund|double charge|billed)\b",
        r"\b(lag|lags|laggy|lagging|stuck|stutter\w*|buffering)\b",
        r"\bsomething went wrong\b",
        r"\b(no sound|no audio|out of sync)\b",
        r"\bfix (this|it|the|my)\b",
        r"\b(greyed|grayed) out\b",
        r"\bnever (loads|works|opens|plays|connects)\b",
        r"\b(don'?t|dont|doesn'?t) want to (open|work|play|load)\b",
    ],
    "question": [
        r"\b(how (do|can|to|does|would)|is there (a|any)|does anyone|anyone (else|know)|can (i|you|someone|anyone)|is it (possible|just me)|what (is|are|happened|does|do)|why (is|does|did|do|are)|where (is|are|can|did)|which|should i|has anyone|do you|am i)\b[^.!]*\?",
        r"^\s*(how|what|why|where|is|does|do|can|has|anyone|which|am|are)\b[^.!]*\?",
        r"\?\s*$",
    ],
    "complaint": [
        r"\b(worst|terrible|trash|garbage|hate|annoying|awful|horrible|ridiculous|useless|pathetic|disgusting|sucks?|ruin\w*|disappoint\w*|frustrat\w*|unusable|joke|greedy|scam|overpriced|expensive)\b",
        r"\b(worse|bad|poor|stupid|dumb|lame|mess|annoy\w*|piss\w*|excessive|tired of|enshittification)\b",
        r"\b(too many|so many|tons of|full of|constant) ads\b",
        r"\b(ads?|adds|advertis\w*)\b",
        r"\b(uninstall\w*|cancel\w*|switch\w* to|leaving|unsubscrib\w*)\b",
        r"\b(no longer|not anymore|used to be)\b",
    ],
    "praise": [
        r"\b(love|loving|loved|great|best|amazing|awesome|excellent|perfect|nice|fantastic|wonderful|favorite|favourite|thank\w*|good|cool|enjoy\w*|recommend|happy)\b",
        r"[❤😍🥰👍🔥💯]",
    ],
}

ORDER = ["feature_request", "help_seeking", "bug_report", "question", "complaint", "praise"]
OUTPUT = {"help_seeking": "question"}
_COMPILED = {
    label: [re.compile(p, re.IGNORECASE) for p in pats] for label, pats in PATTERNS.items()
}


def classify(text: str) -> str:
    for label in ORDER:
        if any(p.search(text) for p in _COMPILED[label]):
            return OUTPUT.get(label, label)
    return "other"


def classify_many(texts: Iterable[str]) -> list[str]:
    return [classify(t) for t in texts]


def explain(text: str) -> list[tuple[str, str]]:
    """All (label, matched span) pairs, for error analysis."""
    hits = []
    for label in ORDER:
        for p in _COMPILED[label]:
            if m := p.search(text):
                hits.append((label, m.group(0)))
    return hits
