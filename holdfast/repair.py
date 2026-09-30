"""Detect tool-call argument values that the user appears to have corrected.

This is a *general* speech-repair heuristic, not tied to any benchmark item:
a value counts as a likely "reparandum" when, in the user's transcript, its
last mention is followed closely by a repair marker ("no wait", "actually",
"sorry", "I mean", ...) and further content, and the value is not restated
after the marker.

It is deliberately used only to *challenge* a call once (ask the model to
re-check), never to rewrite arguments, so a false positive costs one extra
model round-trip and never a wrong action.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

# Tokens that carry no content; they do not count toward the marker window.
FILLERS = frozenset(
    {"uh", "um", "umm", "uhm", "er", "erm", "ah", "oh", "hmm", "mm", "like", "okay", "ok", "so", "well", "please"}
)

_MARKER_PHRASES = [
    "on second thought",
    "let me correct",
    "scratch that",
    "change that",
    "change it",
    "make that",
    "make it",
    "or rather",
    "no wait",
    "wait no",
    "no no",
    "not that",
    "hold on",
    "i mean",
    "i meant",
    "my bad",
    "actually",
    "correction",
    "sorry",
    "oops",
    "rather",
    "instead",
    "wait",
    "no",
]
# Longest phrases first so "no wait" wins over "no".
REPAIR_MARKERS: tuple[tuple[str, ...], ...] = tuple(
    sorted((tuple(p.split()) for p in _MARKER_PHRASES), key=len, reverse=True)
)

# Marker followed by these tokens is comparative ("X instead of Y"), not a repair of X.
_COMPARATIVE_NEXT = {("instead",): {"of"}, ("rather",): {"than"}}

# How many content tokens may sit between the value and the marker. In spoken
# repairs the marker follows the corrected value directly ("Lisbon, no wait,
# Porto"); only fillers intervene. Allowing content in between attributes the
# marker to unrelated earlier values (found by scripts/harness_selftest.py).
MARKER_WINDOW = 0


@dataclass(frozen=True)
class Reparandum:
    arg: str
    value: Any
    marker: str
    snippet: str


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,000 -> 1000
    return re.findall(r"[a-z0-9]+", text)


def value_tokens(value: Any) -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    toks = tokenize(str(value))
    content = [t for t in toks if t not in FILLERS]
    if not content or len("".join(content)) < 2:
        return []
    return toks


def _find_all(haystack: list[str], needle: list[str]) -> list[int]:
    n = len(needle)
    return [i for i in range(len(haystack) - n + 1) if haystack[i : i + n] == needle]


def _marker_after(toks: list[str], start: int) -> tuple[int, tuple[str, ...]] | None:
    content_seen = 0
    i = start
    while i < len(toks) and content_seen <= MARKER_WINDOW:  # noqa: SIM
        for mk in REPAIR_MARKERS:
            if tuple(toks[i : i + len(mk)]) == mk:
                nxt = toks[i + len(mk)] if i + len(mk) < len(toks) else None
                if nxt in _COMPARATIVE_NEXT.get(mk, set()):
                    return None
                return i, mk
        if toks[i] not in FILLERS:
            content_seen += 1
        i += 1
    return None


def find_reparanda(text: str, args: dict[str, Any]) -> list[Reparandum]:
    """Return argument values that look corrected. Each marker is attributed
    only to the value that ends closest before it."""
    toks = tokenize(text)
    best: dict[int, tuple[int, Reparandum]] = {}  # marker pos -> (value end, hit)
    for key, value in args.items():
        vt = value_tokens(value)
        if not vt:
            continue
        occurrences = _find_all(toks, vt)
        if not occurrences:
            continue
        last = occurrences[-1]
        found = _marker_after(toks, last + len(vt))
        if found is None:
            continue
        mpos, marker = found
        rest = [t for t in toks[mpos + len(marker) :] if t not in FILLERS]
        if not rest:
            continue
        snippet = " ".join(toks[max(0, last - 3) : min(len(toks), mpos + len(marker) + 5)])
        hit = Reparandum(arg=key, value=value, marker=" ".join(marker), snippet=snippet)
        end = last + len(vt)
        if mpos not in best or end > best[mpos][0]:
            best[mpos] = (end, hit)
    return [hit for _, (_, hit) in sorted(best.items())]


def describe(hits: Iterable[Reparandum]) -> str:
    return "; ".join(f"{h.arg}={h.value!r} (heard: \"...{h.snippet}...\")" for h in hits)
