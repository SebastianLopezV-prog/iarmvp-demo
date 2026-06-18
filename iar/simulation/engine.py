"""Monte Carlo engine (Task 2.3).

The calculation core: roll the price and imbalance "dice" many times, turn each
roll into a settlement cost, and read the risk numbers off the distribution of
those costs.

Pipeline per run
----------------
1. Ask a **draw component** for the random numbers (uniforms) for price and
   imbalance — shape ``(n_scenarios, n_mtus)`` each.
2. Map them through the marginals built earlier:
   - price spread  = ``price_sampler.ppf(u_price)``   (2.2; EUR vs spot)
   - imbalance     = ``imbalance_model.ppf(u_imbalance)`` (2.1; MWh)
3. Build the **per-scenario settlement cost**, summed across all MTUs:
   - absolute imbalance price = ``dam_price + spread``
   - **Gross cost**  = Σ_t  imbalance_t × (dam_price_t + spread_t)
   - **Spread cost** = Σ_t  imbalance_t × spread_t
4. Read the risk numbers off the cost distribution (see "Sign convention").

Sign convention (read this)
---------------------------
``cost`` is a signed euro figure per scenario: **positive = net cost (bad),
negative = net revenue (good)**. It follows directly from
``imbalance = DAM position − delivery``: short (imbalance > 0) at a positive price
costs money; long (imbalance < 0) earns it; negative prices flip both. So the
**worst** outcomes are the *largest* costs — the **upper** tail.

- **IaR** at confidence ``c`` = the ``c``-quantile of summed cost ("we are ``c``
  confident the cost is no worse than this"). Positive ⇒ a loss; negative ⇒ even
  the bad case is still a net revenue.
- **CIaR / Expected Shortfall** = the mean cost across the worst ``1 − c`` tail
  (how ugly it gets once you're past the IaR threshold).

Summed-quantile, NOT sum of per-MTU IaRs
----------------------------------------
We sum each scenario across MTUs *first*, then take the quantile of those whole
horizon totals. We never take a per-MTU quantile and add them up — that would
assume every MTU has its worst draw simultaneously and badly overstate the risk.

Independence + the copula seam
------------------------------
Price and imbalance are sampled **independently** — an explicit MVP simplification
(no dependence modelling). The *only* place dependence would ever enter is the
random-number step, which is isolated behind the swappable :class:`ScenarioDraw`
boundary. The default :class:`IndependentDraw` returns independent uniforms. A
future copula is a single drop-in: a new ``ScenarioDraw`` that returns *jointly
dependent* uniforms (still uniform per margin) and feeds the same ``ppf`` calls —
nothing else in the engine changes. Per scope, **no copula code exists here**;
only the empty socket and the independent implementation.

Pure module: numpy only, no DB. Persistence is Task 2.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from iar.simulation.imbalance_model import ImbalanceModel
from iar.simulation.price_sampler import QuantilePriceSampler


# --------------------------------------------------------------------------- #
# The copula insertion point: the swappable random-number ("draw") step
# --------------------------------------------------------------------------- #
@runtime_checkable
class ScenarioDraw(Protocol):
    """Produces the uniforms that drive the price and imbalance marginals.

    A draw returns ``(u_price, u_imbalance)``, each of shape
    ``(n_scenarios, n_mtus)`` with entries in (0, 1). This is the *single* place
    that price↔imbalance dependence could be introduced: an implementation is
    free to make the two arrays jointly dependent, **as long as each remains
    uniform(0, 1) marginally** so the downstream ``ppf`` mapping stays correct.
    The MVP ships only the independent implementation below.
    """

    def draw(
        self, n_scenarios: int, n_mtus: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]: ...


class IndependentDraw:
    """Independent uniforms for price and imbalance (MVP default — no dependence)."""

    def draw(
        self, n_scenarios: int, n_mtus: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        u_price = rng.random((n_scenarios, n_mtus))
        u_imbalance = rng.random((n_scenarios, n_mtus))
        return u_price, u_imbalance


# --------------------------------------------------------------------------- #
# Configuration and results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EngineConfig:
    """Run controls for the Monte Carlo engine."""

    n_scenarios: int = 10_000
    confidence: float = 0.95
    seed: int | None = 42
    rolling_window_mtus: int = 16  # 4h at 15-min MTUs — for the rolling-window IaR read-off

    def __post_init__(self) -> None:
        if self.n_scenarios <= 0:
            raise ValueError(f"n_scenarios must be > 0, got {self.n_scenarios}")
        if not (0.0 < self.confidence < 1.0):
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.rolling_window_mtus <= 0:
            raise ValueError(f"rolling_window_mtus must be > 0, got {self.rolling_window_mtus}")


@dataclass
class RiskMeasure:
    """IaR/CIaR for one cost basis (gross or spread), EUR.

    ``cost`` is the per-scenario summed cost vector — a diagnostic kept for
    plotting/validation; it is *not* persisted (architecture: store summaries).

    The ``*_per_mtu`` arrays and ``rolling_iar`` are per-MTU / rolling-window
    read-offs of the same scenario set (length ``n_mtus`` / scalar). They power the
    dashboard's intraday, heatmap and per-MTU/rolling limit panels. They are
    *read-offs*, not raw scenarios, so persisting them still honours the
    "store summaries" rule.
    """

    iar: float
    ciar: float
    mean: float
    cost: np.ndarray
    iar_per_mtu: np.ndarray | None = None
    ciar_per_mtu: np.ndarray | None = None
    rolling_iar: float | None = None
    rolling_window: int | None = None

    def summary(self) -> dict[str, float]:
        """Numbers only (no scenario vector) — what gets persisted in 2.4."""
        return {"iar": self.iar, "ciar": self.ciar, "mean": self.mean}

    @property
    def peak_mtu_iar(self) -> float | None:
        """The worst single-MTU IaR (max of the per-MTU series), or ``None``."""
        return None if self.iar_per_mtu is None else float(np.max(self.iar_per_mtu))


@dataclass
class IaRReport:
    """Full engine output: Gross and Spread risk measures + run metadata."""

    confidence: float
    n_scenarios: int
    seed: int | None
    gross: RiskMeasure
    spread: RiskMeasure

    def summary(self) -> dict:
        return {
            "confidence": self.confidence,
            "n_scenarios": self.n_scenarios,
            "seed": self.seed,
            "gross": self.gross.summary(),
            "spread": self.spread.summary(),
        }


# --------------------------------------------------------------------------- #
# Risk read-off
# --------------------------------------------------------------------------- #
def _measure(cost: np.ndarray, confidence: float) -> RiskMeasure:
    """IaR (upper-tail quantile of cost) and CIaR (mean of the worst tail)."""
    iar = float(np.quantile(cost, confidence))
    tail = cost[cost >= iar]
    ciar = float(tail.mean()) if tail.size else iar
    return RiskMeasure(iar=iar, ciar=ciar, mean=float(cost.mean()), cost=cost)


def _per_mtu_measures(cost_mtu: np.ndarray, confidence: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-MTU IaR and CIaR: the standalone risk of each MTU's own cost column.

    NB this is the *opposite* of the summed-quantile period IaR — each MTU is read
    in isolation, so the sum of these per-MTU IaRs overstates the diversified period
    risk (that gap is exactly the dashboard's "overperformance ratio").
    """
    iar = np.quantile(cost_mtu, confidence, axis=0)  # (n_mtus,)
    tail_mask = cost_mtu >= iar[None, :]
    counts = tail_mask.sum(axis=0)
    ciar = np.where(
        counts > 0,
        (cost_mtu * tail_mask).sum(axis=0) / np.maximum(counts, 1),
        iar,
    )
    return iar, ciar


def _rolling_iar(cost_mtu: np.ndarray, confidence: float, window: int) -> float:
    """Worst contiguous ``window``-MTU **summed-quantile** IaR across the horizon.

    For each window position, sum each scenario's cost over the window (preserving
    cross-MTU diversification) and take the quantile; return the worst window. Falls
    back to the whole-horizon period IaR when the window covers all MTUs.
    """
    n = cost_mtu.shape[1]
    w = min(window, n)
    worst = -np.inf
    for start in range(0, n - w + 1):
        q = float(np.quantile(cost_mtu[:, start : start + w].sum(axis=1), confidence))
        worst = max(worst, q)
    return worst


# --------------------------------------------------------------------------- #
# Engine entry point
# --------------------------------------------------------------------------- #
def run_simulation(
    price_sampler: QuantilePriceSampler,
    imbalance_model: ImbalanceModel,
    dam_price: np.ndarray,
    config: EngineConfig | None = None,
    draw: ScenarioDraw | None = None,
) -> IaRReport:
    """Run the Monte Carlo and return Gross/Spread IaR + CIaR.

    Parameters
    ----------
    price_sampler:
        Per-MTU imbalance **spread** marginal (Task 2.2).
    imbalance_model:
        Per-MTU imbalance **volume** distribution in MWh (Task 2.1).
    dam_price:
        Per-MTU day-ahead (spot) price, EUR/MWh, 1-D length ``n_mtus``. Used to
        turn the spread into an absolute price for the Gross basis. (In the MVP
        this is still a stub — see ``load_dam_prices``.)
    config:
        Run controls; defaults to :class:`EngineConfig`.
    draw:
        The random-number component (copula insertion point); defaults to
        :class:`IndependentDraw`.
    """
    config = config or EngineConfig()
    draw = draw or IndependentDraw()

    n_mtus = imbalance_model.n_mtus
    if price_sampler.n_mtus != n_mtus:
        raise ValueError(
            f"MTU mismatch: price_sampler has {price_sampler.n_mtus}, imbalance_model has {n_mtus}"
        )
    dam_price = np.asarray(dam_price, dtype=float)
    if dam_price.shape != (n_mtus,):
        raise ValueError(f"dam_price must be 1-D length {n_mtus}, got shape {dam_price.shape}")
    if not np.all(np.isfinite(dam_price)):
        raise ValueError("dam_price contains non-finite values")

    rng = np.random.default_rng(config.seed)
    u_price, u_imbalance = draw.draw(config.n_scenarios, n_mtus, rng)
    if u_price.shape != (config.n_scenarios, n_mtus) or u_imbalance.shape != (
        config.n_scenarios,
        n_mtus,
    ):
        raise ValueError("draw component returned uniforms of the wrong shape")

    # Marginals -> per-(scenario, MTU) draws.
    spread = price_sampler.ppf(u_price)  # EUR vs spot
    imbalance = imbalance_model.ppf(u_imbalance)  # MWh
    absolute_price = dam_price[None, :] + spread  # EUR/MWh

    # --- Which basis needs the DAM (spot) price? (counterintuitive — read this) ---
    # By definition:
    #   Gross  = imbalance x  imbalance_price                 (total settlement cost)
    #   Spread = imbalance x (imbalance_price - DAM_price)    (under/over-perf vs day-ahead)
    # So DAM appears in the SPREAD definition, not Gross. BUT Optimeering does not
    # publish the absolute imbalance price — it publishes the SPREAD directly, i.e.
    #   spread (s) = imbalance_price - DAM_price.
    # Substituting:
    #   Spread = imbalance x s                 -> uses Optimeering's spread AS-IS;
    #                                             DAM is already baked in, none needed here.
    #   Gross  = imbalance x (DAM_price + s)   -> must ADD DAM back to rebuild the
    #                                             absolute price -> this is why GROSS,
    #                                             not Spread, depends on dam_price.
    # Consequence: Spread IaR's price side is fully live (the spread); Gross additionally
    # needs a real spot-price feed (currently a stub — see load_dam_prices TODO(dam-source)).
    # Per-(scenario, MTU) cost matrices; sum across MTUs for the period total.
    cost_gross_mtu = imbalance * absolute_price
    cost_spread_mtu = imbalance * spread
    gross = _measure(cost_gross_mtu.sum(axis=1), config.confidence)
    spread = _measure(cost_spread_mtu.sum(axis=1), config.confidence)

    # Per-MTU and rolling-window read-offs of the same scenario set (for the
    # intraday / heatmap / per-MTU & rolling limit panels).
    for measure, cost_mtu in ((gross, cost_gross_mtu), (spread, cost_spread_mtu)):
        measure.iar_per_mtu, measure.ciar_per_mtu = _per_mtu_measures(cost_mtu, config.confidence)
        measure.rolling_iar = _rolling_iar(cost_mtu, config.confidence, config.rolling_window_mtus)
        measure.rolling_window = min(config.rolling_window_mtus, n_mtus)

    return IaRReport(
        confidence=config.confidence,
        n_scenarios=config.n_scenarios,
        seed=config.seed,
        gross=gross,
        spread=spread,
    )
