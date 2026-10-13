import numpy as np
import pytest

from tokenizer_tax.measure import measure, tax_with_ci


def test_equal_counts_mean_zero_tax():
    tax, lo, hi = tax_with_ci(np.array([3, 5, 7]), np.array([3, 5, 7]))
    assert tax == 0 and lo == 0 and hi == 0


def test_tax_is_ratio_of_sums_not_mean_of_ratios():
    # 1->2 (+100%) on a short pair, 100->100 on a long one: corpus-level is ~+1%
    tax, _, _ = tax_with_ci(np.array([1, 100]), np.array([2, 100]))
    assert tax == pytest.approx(1 / 101)


def test_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    en = rng.integers(5, 40, size=300)
    fr = (en * 1.3).astype(int)
    tax, lo, hi = tax_with_ci(en, fr)
    assert lo <= tax <= hi


def test_rejects_mismatched_input():
    with pytest.raises(ValueError):
        tax_with_ci(np.array([1, 2]), np.array([1]))


def test_measure_with_whitespace_counter():
    count = lambda s: len(s.split())
    r = measure(count, [("the cat", "le petit chat"), ("hi", "salut")])
    assert (r.tokens_en, r.tokens_fr) == (3, 4)
    assert r.tax == pytest.approx(1 / 3)
