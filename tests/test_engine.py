"""Engine validation tests (Task 2.3 / seeds of 2.5).

Covers: an analytic all-Normal case (closed-form IaR), Monte Carlo convergence,
seed reproducibility, the "summed-quantile NOT sum of per-MTU IaRs" rule,
price/imbalance independence, and that the copula insertion point is a real
swappable seam (without any copula being implemented).
"""

import numpy as np
import pytest
from scipy import stats

from iar.simulation.engine import (
    EngineConfig,
    IndependentDraw,
    ScenarioDraw,
    run_simulation,
)
from iar.simulation.imbalance_model import ImbalanceModel
from iar.simulation.price_sampler import QuantilePriceSampler


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def const_price(values_per_mtu):
    """A degenerate price sampler returning a fixed spread per MTU (ppf == const)."""
    vals = np.asarray(values_per_mtu, dtype=float)
    levels = np.array([0.25, 0.5, 0.75])
    matrix = np.repeat(vals[:, None], 3, axis=1)  # equal across levels -> constant
    return QuantilePriceSampler(levels, matrix)


def normal_imbalance(mu, sigma):
    return ImbalanceModel(np.asarray(mu, float), np.asarray(sigma, float), dist="normal")


# --------------------------------------------------------------------------- #
# Analytic all-Normal case
# --------------------------------------------------------------------------- #
def test_analytic_normal_iar_matches_closed_form():
    # Deterministic price -> cost is a sum of independent normals (closed form).
    mu = np.array([1.0, 1.5, 2.0, 0.5, 1.2])
    sigma = np.full(5, 2.0)
    spread = np.full(5, 5.0)
    dam = np.full(5, 40.0)
    k_gross = dam + spread  # 45 each

    rep = run_simulation(
        const_price(spread),
        normal_imbalance(mu, sigma),
        dam_price=dam,
        config=EngineConfig(n_scenarios=300_000, confidence=0.95, seed=7),
    )

    z = stats.norm.ppf(0.95)
    mean_g = float(np.sum(k_gross * mu))
    sd_g = float(np.sqrt(np.sum((k_gross * sigma) ** 2)))
    iar_analytic = mean_g + z * sd_g
    assert rep.gross.iar == pytest.approx(iar_analytic, rel=0.02)

    # Spread basis uses k = spread only.
    mean_s = float(np.sum(spread * mu))
    sd_s = float(np.sqrt(np.sum((spread * sigma) ** 2)))
    assert rep.spread.iar == pytest.approx(mean_s + z * sd_s, rel=0.02)


def test_ciar_is_worse_than_iar_and_matches_normal_es():
    mu = np.array([2.0, 2.0, 2.0])
    sigma = np.full(3, 3.0)
    spread = np.full(3, 0.0)  # spread basis: k = 1*... use dam only for gross
    dam = np.full(3, 10.0)
    rep = run_simulation(
        const_price(np.full(3, 1.0)),  # constant spread = 1
        normal_imbalance(mu, sigma),
        dam_price=dam,
        config=EngineConfig(n_scenarios=300_000, confidence=0.95, seed=3),
    )
    assert rep.gross.ciar > rep.gross.iar  # tail mean is beyond the threshold
    # closed-form ES for a normal: mean + sd * pdf(z)/(1-c)
    k = dam + 1.0
    mean_g = float(np.sum(k * mu))
    sd_g = float(np.sqrt(np.sum((k * sigma) ** 2)))
    z = stats.norm.ppf(0.95)
    es = mean_g + sd_g * stats.norm.pdf(z) / 0.05
    assert rep.gross.ciar == pytest.approx(es, rel=0.03)


# --------------------------------------------------------------------------- #
# Convergence + reproducibility
# --------------------------------------------------------------------------- #
def test_convergence_error_shrinks_with_more_scenarios():
    mu = np.array([1.0, 2.0, 1.5])
    sigma = np.full(3, 2.0)
    spread = np.full(3, 3.0)
    dam = np.full(3, 20.0)
    k = dam + spread
    z = stats.norm.ppf(0.95)
    target = float(np.sum(k * mu)) + z * float(np.sqrt(np.sum((k * sigma) ** 2)))

    def avg_err(n):
        errs = []
        for seed in range(6):
            rep = run_simulation(
                const_price(spread), normal_imbalance(mu, sigma), dam_price=dam,
                config=EngineConfig(n_scenarios=n, confidence=0.95, seed=seed),
            )
            errs.append(abs(rep.gross.iar - target))
        return float(np.mean(errs))

    assert avg_err(50_000) < avg_err(500)


def test_seed_reproducibility():
    mu, sigma = np.array([1.0, -2.0, 0.5]), np.full(3, 2.0)
    price = QuantilePriceSampler(
        np.array([0.1, 0.5, 0.9]), np.tile([-5.0, 0.0, 8.0], (3, 1))
    )
    args = (price, normal_imbalance(mu, sigma))
    cfg = EngineConfig(n_scenarios=20_000, seed=99)
    a = run_simulation(*args, dam_price=np.full(3, 30.0), config=cfg)
    b = run_simulation(*args, dam_price=np.full(3, 30.0), config=cfg)
    assert a.gross.iar == b.gross.iar and a.spread.iar == b.spread.iar
    np.testing.assert_array_equal(a.gross.cost, b.gross.cost)


# --------------------------------------------------------------------------- #
# Summed-quantile, NOT sum of per-MTU IaRs
# --------------------------------------------------------------------------- #
def test_summed_quantile_is_below_sum_of_per_mtu_iars():
    # Diversification: quantile of the sum < sum of the per-MTU quantiles.
    mu = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    sigma = np.full(5, 2.0)
    spread = np.full(5, 4.0)
    dam = np.full(5, 30.0)
    k = dam + spread
    z = stats.norm.ppf(0.95)

    rep = run_simulation(
        const_price(spread), normal_imbalance(mu, sigma), dam_price=dam,
        config=EngineConfig(n_scenarios=300_000, confidence=0.95, seed=11),
    )
    # Naive (wrong) method: sum of per-MTU IaRs = mean + z * sum_t |k_t| sigma_t.
    naive_sum = float(np.sum(k * mu)) + z * float(np.sum(np.abs(k) * sigma))
    engine_iar = rep.gross.iar
    correct_summed = float(np.sum(k * mu)) + z * float(np.sqrt(np.sum((k * sigma) ** 2)))

    assert engine_iar == pytest.approx(correct_summed, rel=0.02)
    assert engine_iar < naive_sum  # the engine does NOT overstate risk


# --------------------------------------------------------------------------- #
# Independence + the copula seam
# --------------------------------------------------------------------------- #
def test_independent_draw_is_uncorrelated_and_uniform():
    up, ui = IndependentDraw().draw(200_000, 4, np.random.default_rng(0))
    assert up.shape == ui.shape == (200_000, 4)
    for c in range(4):
        assert abs(np.corrcoef(up[:, c], ui[:, c])[0, 1]) < 0.02
    assert 0.0 < up.min() and up.max() < 1.0


def test_independent_draw_satisfies_protocol():
    assert isinstance(IndependentDraw(), ScenarioDraw)


def test_copula_seam_is_swappable_without_a_copula_impl():
    # Test-only stub (NOT a copula): identical uniforms -> perfect dependence.
    class _ComonotonicDraw:
        def draw(self, n, m, rng):
            u = rng.random((n, m))
            return u, u

    price = QuantilePriceSampler(
        np.array([0.05, 0.5, 0.95]), np.tile([-20.0, 0.0, 30.0], (4, 1))
    )
    imb = normal_imbalance(np.full(4, 1.0), np.full(4, 3.0))
    dam = np.full(4, 25.0)
    cfg = EngineConfig(n_scenarios=200_000, seed=5)

    indep = run_simulation(price, imb, dam_price=dam, config=cfg, draw=IndependentDraw())
    dep = run_simulation(price, imb, dam_price=dam, config=cfg, draw=_ComonotonicDraw())
    # Swapping the draw component changes the result -> the seam is real & wired.
    assert abs(indep.gross.iar - dep.gross.iar) > 1.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kwargs", [{"n_scenarios": 0}, {"confidence": 1.0}, {"confidence": 0.0}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)


def test_mtu_mismatch_rejected():
    price = QuantilePriceSampler(np.array([0.25, 0.75]), np.zeros((3, 2)))  # 3 MTUs
    imb = normal_imbalance(np.zeros(5), np.ones(5))  # 5 MTUs
    with pytest.raises(ValueError, match="MTU mismatch"):
        run_simulation(price, imb, dam_price=np.zeros(5))


def test_dam_price_shape_rejected():
    price = QuantilePriceSampler(np.array([0.25, 0.75]), np.zeros((3, 2)))
    imb = normal_imbalance(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="dam_price must be 1-D length 3"):
        run_simulation(price, imb, dam_price=np.zeros(4))
