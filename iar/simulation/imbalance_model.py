"""Imbalance-uncertainty model (Task 2.1).

Turns the *expected* per-MTU portfolio imbalance into a sampleable probability
distribution, so the Monte Carlo engine (2.3) can draw imbalance scenarios.

Domain
------
Portfolio imbalance (MWh per MTU) = DAM position − actual delivery. Actual
delivery is uncertain because metered generation deviates from forecast::

    delivery_t      = forecast_gen_t + error_t
    imbalance_t     = dam_t − delivery_t
                    = (dam_t − forecast_gen_t) − error_t
                    = expected_imbalance_t − error_t

So the imbalance distribution is centred on the **expected imbalance**
``expected_imbalance_t = dam_t − forecast_gen_t`` and its spread comes entirely
from the generation forecast error ``error_t``. Because the error is symmetric,
the imbalance is distributed as ``mean = expected_imbalance_t`` with standard
deviation ``sigma_t``.

Why parametric (and swappable)
------------------------------
We have no historical forecast-error data yet, so ``sigma_t`` is modelled
parametrically as a configurable fraction of either installed **capacity** or the
per-MTU **forecast** generation (see ``ImbalanceModelConfig``). Two shapes are
offered: a Gaussian (``"normal"``) and a heavier-tailed scaled Student-t
(``"student_t"``). The whole thing is a single small module so the Week-3
backtest can recalibrate ``sigma_fraction`` against realised exceedances by just
re-instantiating with a new config.

Engine interface
----------------
The engine samples price and imbalance **independently** (an explicit MVP
simplification — no copula). To keep the *uniform-draw* step separable (the
documented copula insertion point for a possible future), this model maps
uniforms → imbalance via :meth:`ImbalanceModel.ppf`. :meth:`sample` is a
convenience that draws its own independent uniforms and calls ``ppf``; a future
dependence layer would instead feed correlated uniforms into the same ``ppf``.

This module is pure (numpy/scipy only) — it never touches the DB. Callers pull
arrays from storage and pass them in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

DistName = Literal["normal", "student_t"]
ScaleBasis = Literal["capacity", "forecast"]

# Clip uniforms away from {0, 1} so the inverse-CDF never returns ±inf.
_U_EPS = 1e-9


@dataclass(frozen=True)
class ImbalanceModelConfig:
    """Parameters for the parametric imbalance-uncertainty model.

    Attributes
    ----------
    dist:
        ``"normal"`` (Gaussian) or ``"student_t"`` (heavier tails).
    sigma_fraction:
        Per-MTU standard deviation as a fraction of the chosen ``scale_basis``.
        E.g. ``0.15`` with ``scale_basis="capacity"`` → sigma = 15 % of the
        per-MTU capacity energy.
    scale_basis:
        ``"capacity"`` → sigma is a fraction of installed capacity (same for every
        MTU; robust when forecast generation is near zero). ``"forecast"`` → sigma
        is a fraction of each MTU's forecast generation (proportional uncertainty).
    student_df:
        Degrees of freedom for the Student-t. Must be > 2 so the variance (and
        hence ``sigma`` as a true standard deviation) is finite. Lower = heavier
        tails. Ignored when ``dist="normal"``.
    sigma_floor_mwh:
        Lower bound on per-MTU sigma (MWh). Useful with ``scale_basis="forecast"``
        so low-wind MTUs keep a non-trivial uncertainty.
    """

    dist: DistName = "normal"
    sigma_fraction: float = 0.15
    scale_basis: ScaleBasis = "capacity"
    student_df: float = 5.0
    sigma_floor_mwh: float = 0.0

    def __post_init__(self) -> None:
        if self.dist not in ("normal", "student_t"):
            raise ValueError(f"dist must be 'normal' or 'student_t', got {self.dist!r}")
        if self.scale_basis not in ("capacity", "forecast"):
            raise ValueError(
                f"scale_basis must be 'capacity' or 'forecast', got {self.scale_basis!r}"
            )
        if self.sigma_fraction <= 0:
            raise ValueError(f"sigma_fraction must be > 0, got {self.sigma_fraction}")
        if self.dist == "student_t" and self.student_df <= 2:
            raise ValueError(
                f"student_df must be > 2 for a finite variance, got {self.student_df}"
            )
        if self.sigma_floor_mwh < 0:
            raise ValueError(f"sigma_floor_mwh must be >= 0, got {self.sigma_floor_mwh}")


class ImbalanceModel:
    """A per-MTU imbalance distribution: mean ``expected_imbalance``, std ``sigma``.

    Construct via :meth:`from_inputs` (the usual path — from DAM position and
    forecast generation) or directly from arrays for testing.
    """

    def __init__(
        self,
        expected_imbalance: np.ndarray,
        sigma: np.ndarray,
        dist: DistName = "normal",
        student_df: float = 5.0,
    ) -> None:
        mean = np.asarray(expected_imbalance, dtype=float)
        sig = np.asarray(sigma, dtype=float)
        if mean.ndim != 1 or sig.ndim != 1:
            raise ValueError("expected_imbalance and sigma must be 1-D arrays")
        if mean.shape != sig.shape:
            raise ValueError(
                f"shape mismatch: expected_imbalance {mean.shape} vs sigma {sig.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sig)):
            raise ValueError("expected_imbalance/sigma contain non-finite values")
        if np.any(sig <= 0):
            raise ValueError("all sigma values must be > 0")

        self._mean = mean
        self._sigma = sig
        self._dist = dist
        self._df = float(student_df)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_inputs(
        cls,
        dam_position: np.ndarray,
        forecast_generation: np.ndarray,
        capacity_mwh: float,
        config: ImbalanceModelConfig | None = None,
    ) -> "ImbalanceModel":
        """Build the model from per-MTU DAM position and forecast generation.

        Parameters
        ----------
        dam_position, forecast_generation:
            1-D arrays (MWh per MTU), aligned and equal length.
        capacity_mwh:
            Per-MTU installed-capacity energy (MW × MTU-hours), used as the sigma
            basis when ``config.scale_basis == "capacity"``.
        config:
            Model configuration; defaults to :class:`ImbalanceModelConfig`.
        """
        config = config or ImbalanceModelConfig()
        dam = np.asarray(dam_position, dtype=float)
        gen = np.asarray(forecast_generation, dtype=float)
        if dam.shape != gen.shape:
            raise ValueError(
                f"dam_position {dam.shape} and forecast_generation {gen.shape} "
                "must have the same shape"
            )
        if dam.ndim != 1:
            raise ValueError("dam_position and forecast_generation must be 1-D")
        if capacity_mwh <= 0:
            raise ValueError(f"capacity_mwh must be > 0, got {capacity_mwh}")

        expected_imbalance = dam - gen

        if config.scale_basis == "capacity":
            sigma = np.full_like(expected_imbalance, config.sigma_fraction * capacity_mwh)
        else:  # "forecast"
            sigma = config.sigma_fraction * np.abs(gen)
        sigma = np.maximum(sigma, config.sigma_floor_mwh)
        # Guard against an all-zero sigma (e.g. forecast basis, zero generation,
        # no floor) which would be a degenerate distribution.
        if np.any(sigma <= 0):
            raise ValueError(
                "computed sigma <= 0 for some MTU; set sigma_floor_mwh > 0 or use "
                "scale_basis='capacity'"
            )

        return cls(expected_imbalance, sigma, config.dist, config.student_df)

    # ------------------------------------------------------------------ #
    # Distribution properties
    # ------------------------------------------------------------------ #
    @property
    def n_mtus(self) -> int:
        return self._mean.shape[0]

    @property
    def mean(self) -> np.ndarray:
        """Per-MTU expected imbalance (the distribution mean), MWh."""
        return self._mean.copy()

    @property
    def sigma(self) -> np.ndarray:
        """Per-MTU standard deviation, MWh."""
        return self._sigma.copy()

    def _scale_param(self) -> np.ndarray:
        """Distribution scale so that the resulting std equals ``sigma``.

        Gaussian: scale = sigma. Student-t has variance ``df/(df-2)``, so to make
        ``sigma`` the actual standard deviation we shrink the scale accordingly.
        """
        if self._dist == "normal":
            return self._sigma
        return self._sigma * np.sqrt((self._df - 2.0) / self._df)

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Inverse CDF (quantile function): map uniforms in (0, 1) to imbalance.

        ``u`` has shape ``(..., n_mtus)``; the last axis aligns with the per-MTU
        mean/sigma. Returns an array of the same shape (MWh). This is the
        separable mapping the engine uses — a future dependence layer would feed
        correlated uniforms here instead of independent ones.
        """
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.n_mtus:
            raise ValueError(
                f"last axis of u ({u.shape[-1]}) must equal n_mtus ({self.n_mtus})"
            )
        u = np.clip(u, _U_EPS, 1.0 - _U_EPS)
        scale = self._scale_param()
        if self._dist == "normal":
            return stats.norm.ppf(u, loc=self._mean, scale=scale)
        return stats.t.ppf(u, self._df, loc=self._mean, scale=scale)

    def sample(self, n_scenarios: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n_scenarios`` independent imbalance vectors.

        Returns an array of shape ``(n_scenarios, n_mtus)`` in MWh. Draws its own
        independent uniforms per MTU and maps them through :meth:`ppf`, so the
        per-MTU marginals are exactly as configured and the MTUs are independent.
        """
        if n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be > 0, got {n_scenarios}")
        rng = rng or np.random.default_rng()
        u = rng.random((n_scenarios, self.n_mtus))
        return self.ppf(u)
