"""Tests for the country-level aggregation core (1.2).

Pure math only - no DB, no network - so they stay hermetic under pytest-socket.
"""

import numpy as np
import pytest

from iar.risk.aggregate import combine_costs, country_name, country_of


def test_country_of_maps_zone_to_prefix():
    assert country_of("SE3") == "SE"
    assert country_of("SE1") == "SE"
    assert country_of("NO2") == "NO"
    assert country_of("se4") == "SE"  # case-insensitive


def test_country_name_friendly_label():
    assert country_name("SE") == "Sweden"
    assert country_name("NO") == "Norway"
    assert country_name("DK") == "DK"  # unknown falls back to the code


def test_combine_costs_quantile_of_sum():
    # Two deterministic vectors: the summed cost is elementwise, then quantiled.
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    iar, ciar = combine_costs([a, b], confidence=0.5)
    total = a + b  # [11, 22, 33, 44]
    assert iar == pytest.approx(np.quantile(total, 0.5))
    assert ciar == pytest.approx(total[total >= iar].mean())


def test_combine_costs_diversifies_independent_zones():
    # Independent draws: the country IaR (quantile of the sum) is strictly below the sum of
    # the per-zone IaRs - that gap is the diversification benefit.
    rng = np.random.default_rng(0)
    z1 = rng.normal(0, 1, 100_000)
    z2 = rng.normal(0, 1, 100_000)
    c = 0.95
    country_iar, _ = combine_costs([z1, z2], confidence=c)
    sum_of_zone_iars = float(np.quantile(z1, c)) + float(np.quantile(z2, c))
    assert country_iar < sum_of_zone_iars
    # Independent equal-variance normals: sum ~ N(0, 2) so quantile ~ sqrt(2) x single.
    assert country_iar == pytest.approx(np.quantile(z1, c) * np.sqrt(2), rel=0.05)


def test_combine_costs_comonotonic_is_the_naive_sum():
    # Identical (perfectly correlated) vectors: no diversification, country == sum of zones.
    v = np.random.default_rng(1).normal(0, 1, 50_000)
    c = 0.95
    iar, _ = combine_costs([v, v], confidence=c)
    assert iar == pytest.approx(2 * np.quantile(v, c))


def test_combine_costs_truncates_to_shortest():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])  # shorter
    iar, _ = combine_costs([a, b], confidence=0.0)  # 0-quantile = min of summed
    assert iar == pytest.approx(11.0)  # min(a[:2]+b) = min(11, 22)


def test_combine_costs_empty():
    assert combine_costs([], confidence=0.95) == (0.0, 0.0)
