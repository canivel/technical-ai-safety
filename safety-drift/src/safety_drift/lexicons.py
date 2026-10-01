"""Register and content lexicons. Used to *measure* the manipulations (reported for every corpus),
never as a behavioural classifier."""

from __future__ import annotations

import re

HEDGE = re.compile(
    r"\b(may|might|could|cannot assure|no assurance|uncertain(?:ty|ties)?|risks?|adversely|caution(?:ary)?|"
    r"potential(?:ly)?|there can be no|unable to|we believe|significant(?:ly)?)\b",
    re.I,
)
SAFETY = re.compile(
    r"\b(cyber ?security|safety|security|threats?|fraud|compliance|protect(?:ion|s|ing)?|breach(?:es)?|"
    r"malware|vulnerabilit(?:y|ies)|hazard(?:s|ous)?|risk management)\b",
    re.I,
)


def per_1k(pattern: re.Pattern, text: str) -> float:
    words = len(text.split())
    return 1000 * len(pattern.findall(text)) / max(words, 1)


def profile(text: str) -> dict:
    return {"hedge_per_1k": per_1k(HEDGE, text), "safety_per_1k": per_1k(SAFETY, text), "words": len(text.split())}

# v2 (review B, C1): harm/threat discussion, excluding the finance senses of "security"/"securities",
# and a hedge lexicon WITHOUT "risk(s)" so register is not measured by topic words.
HARM = re.compile(
    r"\b(cyber[- ]?attacks?|cyber[- ]?security|attacks?|hack(?:ing|ers?|ed)?|malware|ransomware|breach(?:es)?|"
    r"weapons?|injur(?:y|ies|ed)|deaths?|fatal(?:ity|ities)?|explosions?|explosive|toxic|hazard(?:s|ous)?|"
    r"fraud(?:ulent)?|terror(?:ism|ist)?|violen(?:ce|t)|accidents?|contaminat(?:ion|ed)|outbreaks?|pandemic|"
    r"disasters?|fires?|spills?|theft|crim(?:e|es|inal))\b",
    re.I,
)
HEDGE2 = re.compile(
    r"\b(may|might|could|cannot assure|no assurance|uncertain(?:ty|ties)?|adversely|caution(?:ary)?|"
    r"potential(?:ly)?|there can be no|unable to|we believe|significant(?:ly)?)\b",
    re.I,
)
FIRST_PLURAL = re.compile(r"\b(we|our|us)\b", re.I)


def profile_v2(text: str) -> dict:
    return {"harm_per_1k": per_1k(HARM, text), "hedge2_per_1k": per_1k(HEDGE2, text),
            "we_per_1k": per_1k(FIRST_PLURAL, text), "words": len(text.split())}
