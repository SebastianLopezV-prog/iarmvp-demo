"""Price marginal sampler (Task 2.2).

Converts an Optimeering **quantile forecast** into a sampleable per-MTU price
marginal via inverse-CDF (quantile-function) interpolation. The whole point is to
**preserve the shape the forecast actually has** — asymmetry and heavy tails —
rather than collapsing it onto a Normal. So we build the empirical quantile
function directly from the published quantile points and sample from it.

How it works
------------
Optimeering publishes, per MTU (target timestamp), a handful of quantile points
of the imbalance price *spread*, e.g. the PT15M series gives the 9 levels
``1, 5, 10, 25, 50, 75, 90, 95, 99`` (percent) with EUR values. Those points
define a step-wise picture of the CDF; we connect them into a continuous,
monotone **quantile function** ``Q(u)`` by linear interpolation in
(probability, value) space. Sampling is then just ``Q(U)`` for uniforms ``U`` —
the standard inverse-CDF method. Because we interpolate the real points, the
sampled distribution reproduces the forecast's skew and fat tails.

Tails beyond the outermost quantiles
-------------------------------------
The forecast says nothing below P01 or above P99 (1 % of mass each). Two policies
(``tail=``):

* ``"linear"`` (default) — **extend** each tail with the slope of the outermost
  segment (e.g. P95→P99 for the upper tail). This keeps the tail heavy/asymmetric
  instead of truncating it, which matters for a risk number. No Normal is assumed.
* ``"clamp"`` — hold flat at the P01/P99 values (truncate). Conservative-looking
  but understates extreme imbalance cost; offered mainly for comparison.

Spread vs. absolute price
-------------------------
This sampler is generic: it samples whatever quantile *values* you give it. Feed
it the Optimeering **spread** quantiles and you get spread draws (for Spread IaR).
For Gross IaR the engine shifts by the (deterministic) DAM spot price —
``absolute_price = dam_price + spread`` — either before or after sampling; adding
a per-MTU constant just translates every quantile, so the shape is unchanged.

Engine interface
----------------
Like the imbalance model, the sampler maps uniforms → price via :meth:`ppf`, and
:meth:`sample` draws its own independent uniforms and calls it. Price and
imbalance are sampled **independently** (MVP assumption); the separable
uniform-draw step is kept as the documented (but unused) copula insertion point.

Pure module: numpy only, no DB access. Callers assemble the quantile matrix from
storage and pass it in.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

TailPolicy = Literal["linear", "clamp"]

# Clip uniforms away from {0, 1} so sampling stays finite.
_U_EPS = 1e-9


class QuantilePriceSampler:
    """Per-MTU price marginal defined by quantile points; sampled by inverse-CDF.

    Parameters
    ----------
    quantile_levels:
        1-D array of probability levels in (0, 1), strictly ascending
        (e.g. ``[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]``).
    quantile_values:
        Quantile values, shape ``(n_mtus, n_levels)`` (or 1-D ``(n_levels,)`` for a
        single MTU). Row ``m`` holds the forecast quantiles for MTU ``m`` at the
        corresponding ``quantile_levels``. Each row must be non-decreasing across
        levels; tiny crossings from forecast noise are repaired automatically.
    tail:
        Tail-extension policy beyond the outermost quantiles — ``"linear"``
        (default, preserves tails) or ``"clamp"`` (truncate).
    """

    def __init__(
        self,
        quantile_levels: np.ndarray,
        quantile_values: np.ndarray,
        tail: TailPolicy = "linear",
    ) -> None:
        levels = np.asarray(quantile_levels, dtype=float)
        values = np.asarray(quantile_values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if levels.ndim != 1 or levels.size < 2:
            raise ValueError("quantile_levels must be 1-D with at least 2 levels")
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError("quantile_levels must lie strictly in (0, 1)")
        if np.any(np.diff(levels) <= 0):
            raise ValueError("quantile_levels must be strictly ascending")
        if values.ndim != 2 or values.shape[1] != levels.size:
            raise ValueError(
                f"quantile_values must be (n_mtus, {levels.size}); got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("quantile_values contain non-finite entries")
        if tail not in ("linear", "clamp"):
            raise ValueError(f"tail must be 'linear' or 'clamp', got {tail!r}")

        # Repair minor non-monotonicity (forecast noise) so Q(u) is valid.
        self._levels = levels
        self._values = np.maximum.accumulate(values, axis=1)
        self._tail = tail

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_percentiles(
        cls,
        percentile_levels: np.ndarray,
        quantile_values: np.ndarray,
        tail: TailPolicy = "linear",
    ) -> "QuantilePriceSampler":
        """Build from percentile levels (e.g. ``[1, 5, ..., 99]``) instead of (0, 1)."""
        return cls(np.asarray(percentile_levels, dtype=float) / 100.0, quantile_values, tail)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def n_mtus(self) -> int:
        return self._values.shape[0]

    @property
    def n_levels(self) -> int:
        return self._levels.shape[0]

    @property
    def levels(self) -> np.ndarray:
        return self._levels.copy()

    # ------------------------------------------------------------------ #
    # Inverse CDF + sampling
    # ------------------------------------------------------------------ #
    def ppf(self, u: np.ndarray) -> np.ndarray:
        """Quantile function: map uniforms in (0, 1) to price per MTU.

        ``u`` has shape ``(..., n_mtus)`` (last axis aligns with MTUs). Returns the
        same shape. Piecewise-linear between quantile knots; tails handled per the
        ``tail`` policy. This is the separable mapping the engine drives.
        """
        u = np.asarray(u, dtype=float)
        if u.shape[-1] != self.n_mtus:
            raise ValueError(
                f"last axis of u ({u.shape[-1]}) must equal n_mtus ({self.n_mtus})"
            )
        u = np.clip(u, _U_EPS, 1.0 - _U_EPS)
        if self._tail == "clamp":
            # Restrict to the supported range -> flat tails (truncation).
            u = np.clip(u, self._levels[0], self._levels[-1])

        levels = self._levels
        L = levels.size
        # Bracketing segment [k, k+1]; clip to interior so the outer segments are
        # reused for any u outside [levels[0], levels[-1]] -> linear extrapolation.
        k = np.clip(np.searchsorted(levels, u, side="right") - 1, 0, L - 2)

        x0 = levels[k]
        x1 = levels[k + 1]
        # Gather per-MTU knot values: result[..., m] uses column m of `values`.
        # Move MTU axis to front for take_along_axis, then move back.
        k_m = np.moveaxis(k, -1, 0)              # (n_mtus, ...)
        flat_idx = k_m.reshape(self.n_mtus, -1)  # (n_mtus, K)
        y0 = np.take_along_axis(self._values, flat_idx, axis=1).reshape(k_m.shape)
        y1 = np.take_along_axis(self._values, flat_idx + 1, axis=1).reshape(k_m.shape)
        y0 = np.moveaxis(y0, 0, -1)              # back to (..., n_mtus)
        y1 = np.moveaxis(y1, 0, -1)

        slope = (y1 - y0) / (x1 - x0)
        return y0 + (u - x0) * slope

    def sample(self, n_scenarios: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Draw ``n_scenarios`` independent price vectors, shape ``(n_scenarios, n_mtus)``."""
        if n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be > 0, got {n_scenarios}")
        rng = rng or np.random.default_rng()
        u = rng.random((n_scenarios, self.n_mtus))
        return self.ppf(u)

    def quantile(self, prob: float) -> np.ndarray:
        """Per-MTU value at a single probability ``prob`` in (0, 1)."""
        return self.ppf(np.full(self.n_mtus, float(prob)))
