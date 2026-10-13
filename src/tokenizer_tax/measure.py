"""The measurement: how many more tokens does French cost than English?

tax = sum(tokens_fr) / sum(tokens_en) - 1, at corpus level (a ratio of sums, so
long sentences weigh more, as they do on a bill). Uncertainty is a percentile
bootstrap over sentence pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TaxResult:
    tokens_en: int
    tokens_fr: int
    tax: float  # e.g. 0.25 means French costs 25% more tokens
    ci_low: float
    ci_high: float
    chars_per_token_en: float
    chars_per_token_fr: float


def tax_with_ci(
    en_counts: np.ndarray,
    fr_counts: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Corpus-level tax and 95% percentile-bootstrap CI over pairs."""
    en_counts = np.asarray(en_counts, dtype=float)
    fr_counts = np.asarray(fr_counts, dtype=float)
    if en_counts.shape != fr_counts.shape or en_counts.size == 0:
        raise ValueError("need two equal-length, non-empty count arrays")
    point = fr_counts.sum() / en_counts.sum() - 1
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, en_counts.size, size=(n_boot, en_counts.size))
    boots = fr_counts[idx].sum(axis=1) / en_counts[idx].sum(axis=1) - 1
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def measure(count, pairs: list[tuple[str, str]], **kw) -> TaxResult:
    en = np.array([count(e) for e, _ in pairs])
    fr = np.array([count(f) for _, f in pairs])
    tax, lo, hi = tax_with_ci(en, fr, **kw)
    chars_en = sum(len(e) for e, _ in pairs)
    chars_fr = sum(len(f) for _, f in pairs)
    return TaxResult(
        tokens_en=int(en.sum()),
        tokens_fr=int(fr.sum()),
        tax=tax,
        ci_low=lo,
        ci_high=hi,
        chars_per_token_en=chars_en / en.sum(),
        chars_per_token_fr=chars_fr / fr.sum(),
    )
