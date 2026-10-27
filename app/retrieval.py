"""Lexical retrieval over the policy chunks: TF-IDF on words plus character n-grams.

It runs offline in milliseconds and needs no model download. The character n-grams
catch spelling variants ("hospitalisation" / "hospitalization", "charge" / "charges").
It does not understand paraphrases the way an embedding model would; see the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from app.documents import Chunk

TOKEN_RE = re.compile(r"\w+")
NUMBER_WORDS = {
    "one": "1", "first": "1", "two": "2", "second": "2", "three": "3", "third": "3",
    "four": "4", "fourth": "4", "five": "5", "fifth": "5", "six": "6", "ten": "10",
    "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
}  # fmt: skip


def normalize(text: str) -> str:
    """Lowercase, map number words to digits, keep single digits, drop stop words."""
    tokens = (NUMBER_WORDS.get(t, t) for t in TOKEN_RE.findall(text.lower()))
    return " ".join(t for t in tokens if t not in ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("no document chunks to index")
        self.chunks = chunks
        docs = [normalize(f"{c.doc_title} {c.section} {c.text}") for c in chunks]
        self._word = TfidfVectorizer(token_pattern=r"\b\w+\b", ngram_range=(1, 2), sublinear_tf=True)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._word_matrix = self._word.fit_transform(docs)
        self._char_matrix = self._char.fit_transform(docs)

    def _scores(self, query: str, word_matrix, char_matrix) -> np.ndarray:
        q = normalize(query)
        word = (word_matrix @ self._word.transform([q]).T).toarray().ravel()
        char = (char_matrix @ self._char.transform([q]).T).toarray().ravel()
        return 0.5 * word + 0.5 * char

    def search(self, query: str, k: int = 4) -> list[Hit]:
        scores = self._scores(query, self._word_matrix, self._char_matrix)
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self.chunks[i], round(float(scores[i]), 4)) for i in order if scores[i] > 0]

    def score_lines(self, query: str, lines: list[str] | tuple[str, ...]) -> np.ndarray:
        """Similarity of the query to individual lines (used by the extractive answerer)."""
        normalized = [normalize(line) for line in lines]
        return self._scores(query, self._word.transform(normalized), self._char.transform(normalized))
