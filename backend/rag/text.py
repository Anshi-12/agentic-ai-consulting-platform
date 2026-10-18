"""Tokenization and chunking utilities (no external NLP dependencies)."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being below between both but
by can could did do does doing down during each few for from further had has have having he her here hers herself
him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once only or
other our ours ourselves out over own same she should so some such than that the their theirs them themselves then
there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves within also may might must shall us via per
""".split())


def stem(token: str) -> str:
    """Very light suffix stripping so 'routers'/'router' and 'failures'/'failure' match."""
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es") and token[-3] in "sxz":
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss") and not token[-2].isdigit():
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS and len(t) > 1]


def clean_text(text: str) -> str:
    text = (text or "").replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\n(?=[a-z])", "", text)        # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9“\"(])")


def split_sentences(text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text)
    return [s.strip() for s in _SENT.split(flat) if len(s.strip()) > 1]


def chunk_text(text: str, max_chars: int = 1100, overlap_chars: int = 180) -> list[str]:
    """Sentence-aware chunking with character overlap."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sentences:
        if len(s) > max_chars:  # very long "sentence" (tables): hard split
            for i in range(0, len(s), max_chars - overlap_chars):
                piece = s[i : i + max_chars]
                if cur:
                    chunks.append(" ".join(cur))
                    cur, size = [], 0
                chunks.append(piece)
            continue
        if size + len(s) > max_chars and cur:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            t = 0
            for prev in reversed(cur):
                if t + len(prev) > overlap_chars:
                    break
                tail.insert(0, prev)
                t += len(prev)
            cur, size = tail, t
        cur.append(s)
        size += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if len(c) >= 80]
