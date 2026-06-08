"""Tests for the quantile price sampler (Task 2.2)."""

import numpy as np
import pytest

from iar.simulation.price_sampler import QuantilePriceSampler

# Optimeering PT15M levels (as probabilities).
LEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
# An asymmetric / right-skewed spread row (heavy upper tail), EUR.
SKEW = np.array([-30.0, -18.0, -12.0, -5.0, -1.0, 4.0, 15.0, 28.0, 80.0])


def test_ppf_passes_through_the_quantile_knots():
    s = QuantilePriceSampler(LEVELS, SKEW)
    # Q(level_i) must return value_i exactly (interpolation hits the knots).
    got = s.ppf(LEVELS.reshape(-1, 1))  # (9,1): one MTU, evaluate at each level
    np.testing.assert_allclose(got.ravel(), SKEW, atol=1e-9)


def test_quantile_helper_and_median():
    s = QuantilePriceSampler(LEVELS, SKEW)
    np.testing.assert_allclose(s.quantile(0.50), [-1.0], atol=1e-9)


def test_ppf_is_monotonic_in_u():
    s = QuantilePriceSampler(LEVELS, SKEW)
    u = np.linspace(0.001, 0.999, 200).reshape(-1, 1)
    q = s.ppf(u).ravel()
    assert np.all(np.diff(q) >= -1e-12)


def test_sample_shape_and_reproducibility():
    vals = np.vstack([SKEW, SKEW + 5.0, SKEW - 3.0])  # 3 MTUs
    s = QuantilePriceSampler(LEVELS, vals)
    a = s.sample(5000, rng=np.random.default_rng(0))
    b = s.sample(5000, rng=np.random.default_rng(0))
    assert a.shape == (5000, 3)
    np.testing.assert_array_equal(a, b)


def test_empirical_quantiles_match_input():
    # Inverse-CDF sampling must reproduce the input quantiles (not a refit shape).
    s = QuantilePriceSampler(LEVELS, SKEW)
    draws = s.sample(400_000, rng=np.random.default_rng(1)).ravel()
    for lvl, val in zip(LEVELS, SKEW):
        # skip the extrapolated extremes; check the interior knots
        if 0.05 <= lvl <= 0.95:
            assert abs(np.quantile(draws, lvl) - val) < 1.5


def test_preserves_asymmetry_not_normal():
    # A right-skewed forecast must yield right-skewed samples (mean > median),
    # i.e. it is NOT being symmetrised into a Normal.
    s = QuantilePriceSampler(LEVELS, SKEW)
    draws = s.sample(200_000, rng=np.random.default_rng(2)).ravel()
    assert draws.mean() > np.median(draws) + 1.0


def test_linear_tail_extends_beyond_outer_knots():
    s = QuantilePriceSampler(LEVELS, SKEW, tail="linear")
    draws = s.sample(200_000, rng=np.random.default_rng(3)).ravel()
    # Upper tail should be able to exceed the P99 knot (80.0); lower below P01.
    assert draws.max() > 80.0
    assert draws.min() < -30.0


def test_clamp_tail_truncates_at_outer_knots():
    s = QuantilePriceSampler(LEVELS, SKEW, tail="clamp")
    draws = s.sample(200_000, rng=np.random.default_rng(4)).ravel()
    assert draws.max() <= 80.0 + 1e-9
    assert draws.min() >= -30.0 - 1e-9


def test_non_monotone_rows_are_repaired():
    bad = SKEW.copy()
    bad[5] = bad[4] - 2.0  # introduce a small crossing
    s = QuantilePriceSampler(LEVELS, bad)
    # repaired to non-decreasing
    assert np.all(np.diff(s.ppf(LEVELS.reshape(-1, 1)).ravel()) >= -1e-12)


def test_from_percentiles_matches_probability_form():
    pct = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99], dtype=float)
    s1 = QuantilePriceSampler.from_percentiles(pct, SKEW)
    s2 = QuantilePriceSampler(LEVELS, SKEW)
    np.testing.assert_allclose(s1.levels, s2.levels)


@pytest.mark.parametrize(
    "levels,values",
    [
        (np.array([0.5]), np.array([[1.0]])),                  # <2 levels
        (np.array([0.0, 0.5, 1.0]), np.zeros((1, 3))),         # levels at 0/1
        (np.array([0.5, 0.25, 0.75]), np.zeros((1, 3))),       # not ascending
        (np.array([0.1, 0.9]), np.zeros((1, 3))),              # shape mismatch
    ],
)
def test_validation_rejects_bad_inputs(levels, values):
    with pytest.raises(ValueError):
        QuantilePriceSampler(levels, values)


def test_bad_tail_rejected():
    with pytest.raises(ValueError):
        QuantilePriceSampler(LEVELS, SKEW, tail="gaussian")


def test_ppf_wrong_width_rejected():
    s = QuantilePriceSampler(LEVELS, np.vstack([SKEW, SKEW]))  # 2 MTUs
    with pytest.raises(ValueError, match="must equal n_mtus"):
        s.ppf(np.zeros((10, 3)))
