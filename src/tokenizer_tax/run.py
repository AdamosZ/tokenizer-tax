"""Run every tokenizer on every corpus; write results/results.csv and the chart.

    python -m tokenizer_tax.run
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .data import CORPORA
from .measure import measure, tax_with_ci
from .tokenizers import REGISTRY, load

RESULTS = Path(__file__).resolve().parents[2] / "results"

FIELDS = [
    "corpus", "tokenizer", "pairs", "tokens_en", "tokens_fr",
    "tax", "ci_low", "ci_high", "chars_per_token_en", "chars_per_token_fr",
]


def char_baseline(pairs):
    """How much longer French is in raw characters: the part of the tax that is
    the language, not the tokenizer."""
    en = np.array([len(e) for e, _ in pairs])
    fr = np.array([len(f) for _, f in pairs])
    return tax_with_ci(en, fr)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chart", action="store_true")
    args = ap.parse_args(argv)

    RESULTS.mkdir(exist_ok=True)
    rows = []
    for corpus_name, loader in CORPORA.items():
        pairs = loader()
        tax, lo, hi = char_baseline(pairs)
        rows.append(dict(corpus=corpus_name, tokenizer="(characters)", pairs=len(pairs),
                         tokens_en="", tokens_fr="", tax=round(tax, 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4),
                         chars_per_token_en="", chars_per_token_fr=""))
        for tok_name in REGISTRY:
            r = measure(load(tok_name).count, pairs)
            rows.append(dict(
                corpus=corpus_name, tokenizer=tok_name, pairs=len(pairs),
                tokens_en=r.tokens_en, tokens_fr=r.tokens_fr,
                tax=round(r.tax, 4), ci_low=round(r.ci_low, 4), ci_high=round(r.ci_high, 4),
                chars_per_token_en=round(r.chars_per_token_en, 3),
                chars_per_token_fr=round(r.chars_per_token_fr, 3),
            ))
            print(f"{corpus_name:34s} {tok_name:24s} tax {r.tax:+.1%} "
                  f"[{r.ci_low:+.1%}, {r.ci_high:+.1%}]")

    out = RESULTS / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    if not args.no_chart:
        from .chart import draw
        draw(rows, RESULTS / "chart.png")
        print(f"wrote {RESULTS / 'chart.png'}")


if __name__ == "__main__":
    main()
