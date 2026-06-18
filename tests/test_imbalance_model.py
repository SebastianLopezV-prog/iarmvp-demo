"""Tests for the imbalance-uncertainty model (Task 2.1)."""

import numpy as np
import pytest

from iar.simulation.imbalance_model import ImbalanceModel, ImbalanceModelConfig


def test_mean_is_dam_minus_forecast():
    dam = np.array([10.0, 8.0, 12.0])
    gen = np.array([9.0, 9.5, 7.0])
    m = ImbalanceModel.from_inputs(dam, gen, capacity_mwh=25.0)
    np.testing.assert_allclose(m.mean, dam - gen)


def test_sigma_capacity_basis_is_constant_fraction():
    dam = np.zeros(4)
    gen = np.array([1.0, 5.0, 10.0, 20.0])
    cfg = ImbalanceModelConfig(sigma_fraction=0.2, scale_basis="capacity")
    m = ImbalanceModel.from_inputs(dam, gen, capacity_mwh=25.0, config=cfg)
    np.testing.assert_allclose(m.sigma, np.full(4, 0.2 * 25.0))


def test_sigma_forecast_basis_scales_with_forecast_and_respects_floor():
    gen = np.array([0.0, 10.0, 50.0])
    cfg = ImbalanceModelConfig(sigma_fraction=0.1, scale_basis="forecast", sigma_floor_mwh=1.0)
    m = ImbalanceModel.from_inputs(np.zeros(3), gen, capacity_mwh=100.0, config=cfg)
    # 0.1*[0,10,50] = [0,1,5] -> floored at 1 -> [1,1,5]
    np.testing.assert_allclose(m.sigma, np.array([1.0, 1.0, 5.0]))


def test_sample_shape_and_moments_normal():
    n = 6
    dam = np.linspace(-5, 5, n)
    gen = np.zeros(n)
    cfg = ImbalanceModelConfig(sigma_fraction=0.1, scale_basis="capacity")
    m = ImbalanceModel.from_inputs(dam, gen, capacity_mwh=50.0, config=cfg)  # sigma=5
    draws = m.sample(50_000, rng=np.random.default_rng(0))
    assert draws.shape == (50_000, n)
    np.testing.assert_allclose(draws.mean(axis=0), m.mean, atol=0.1)
    np.testing.assert_allclose(draws.std(axis=0), m.sigma, rtol=0.05)


def test_student_t_std_matches_sigma():
    # The scaled-t must have std == sigma (scale shrunk by sqrt((df-2)/df)).
    cfg = ImbalanceModelConfig(
        dist="student_t", student_df=5.0, sigma_fraction=0.1, scale_basis="capacity"
    )
    m = ImbalanceModel.from_inputs(np.zeros(3), np.zeros(3), capacity_mwh=100.0, config=cfg)
    draws = m.sample(200_000, rng=np.random.default_rng(1))
    np.testing.assert_allclose(draws.std(axis=0), m.sigma, rtol=0.05)  # sigma=10


def test_ppf_median_equals_mean_and_is_monotonic():
    m = ImbalanceModel(np.array([2.0, -3.0]), np.array([1.0, 4.0]))
    # median (u=0.5) of a symmetric dist == mean
    np.testing.assert_allclose(m.ppf(np.array([0.5, 0.5])), m.mean, atol=1e-9)
    lo = m.ppf(np.array([0.1, 0.1]))
    hi = m.ppf(np.array([0.9, 0.9]))
    assert np.all(hi > lo)


def test_sample_is_seed_reproducible():
    m = ImbalanceModel.from_inputs(np.array([1.0, 2.0]), np.array([0.0, 0.0]), capacity_mwh=10.0)
    a = m.sample(1000, rng=np.random.default_rng(42))
    b = m.sample(1000, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_ppf_accepts_engine_supplied_uniforms():
    # The engine owns the uniform draw (copula insertion point); ppf maps them.
    m = ImbalanceModel.from_inputs(np.zeros(3), np.zeros(3), capacity_mwh=10.0)
    u = np.random.default_rng(7).random((100, 3))
    out = m.ppf(u)
    assert out.shape == (100, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dist": "weird"},
        {"scale_basis": "nope"},
        {"sigma_fraction": 0.0},
        {"dist": "student_t", "student_df": 2.0},
        {"sigma_floor_mwh": -1.0},
    ],
)
def test_config_validation_rejects_bad_params(kwargs):
    with pytest.raises(ValueError):
        ImbalanceModelConfig(**kwargs)


def test_ppf_wrong_width_rejected():
    m = ImbalanceModel(np.zeros(3), np.ones(3))
    with pytest.raises(ValueError, match="must equal n_mtus"):
        m.ppf(np.zeros((10, 2)))
