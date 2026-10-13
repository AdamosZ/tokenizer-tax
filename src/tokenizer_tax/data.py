"""Parallel English/French corpora.

Each corpus is a list of (en, fr) pairs carrying the *same content*, so any
difference in token count is the tokenizer's doing, not the text's.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"

Pair = tuple[str, str]


def wmt14_news() -> list[Pair]:
    """WMT14 newstest2014 fr-en: 3,003 professionally translated news sentences."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "wmt/wmt14", "fr-en/test-00000-of-00001.parquet", repo_type="dataset"
    )
    rows = pd.read_parquet(path)["translation"]
    return [(r["en"], r["fr"]) for r in rows]


def guichet_questions() -> list[Pair]:
    """104 international-student questions about French admin (from the Guichet
    project), original English with an opus-mt French translation."""
    pairs = []
    with open(DATA / "guichet_questions.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            pairs.append((row["en"], row["fr"]))
    return pairs


CORPORA = {
    "news (WMT14, human translation)": wmt14_news,
    "admin questions (Guichet, MT)": guichet_questions,
}
