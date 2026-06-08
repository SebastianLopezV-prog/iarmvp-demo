"""Engine validation report (Task 2.5).

A human-readable demonstration that the Monte Carlo engine is correct. The same
checks run automatically (pass/fail) in tests/test_engine.py; this script PRINTS
the evidence so it can be eyeballed and shown to stakeholders.

Four validations (the ones called for in the plan):
  1. Analytic all-Normal case  -> engine IaR/CIaR match the closed-form numbers.
  2. Convergence vs scenarios  -> the estimate error shrinks ~ 1/sqrt(N).
  3. Seed reproducibility       -> same seed = identical; different seeds = close.
  4. Summed-quantile rule       -> quantile-of-sum < sum-of-per-MTU quantiles
                                   (diversification; the engine does NOT overstate risk).

The trick that makes a closed form exist: make the PRICE deterministic, so the
per-scenario cost = sum_t k_t * imbalance_t is a sum of independent normals, i.e.
itself Normal with a known mean and standard deviation.

Run:  python scripts/validate_engine.py     (exit 0 = all checks passed)
"""

from __future__ import annotations

import sys

import numpy as np
from scipy import stats

from iar.simulation.engine import EngineConfig, run_simulation
from iar.simulation.imbalance_model import ImbalanceModel
from iar.simulation.price_sampler import QuantilePriceSampler

CONF = 0.95
Z = stats.norm.ppf(CONF)

_results: list[bool] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    tag = "[ OK ]" if ok else "[FAIL]"
    print(f"  {tag}  {label}" + (f"  -- {detail}" if detail else ""))
    _results.append(ok)


def const_price(values_per_mtu: np.ndarray) -> QuantilePriceSampler:
    """Degenerate price sampler: ppf returns a fixed spread per MTU."""
    vals = np.asarray(values_per_mtu, float)
    return QuantilePriceSampler(np.array([0.25, 0.5, 0.75]), np.repeat(vals[:, None], 3, axis=1))


def normal_imb(mu: np.ndarray, sigma: np.ndarray) -> ImbalanceModel:
    return ImbalanceModel(np.asarray(mu, float), np.asarray(sigma, float), dist="normal")


def analytic(k: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """Closed-form IaR and CIaR for cost = sum_t k_t * N(mu_t, sigma_t)."""
    mean = float(np.sum(k * mu))
    sd = float(np.sqrt(np.sum((k * sigma) ** 2)))
    iar = mean + Z * sd
    ciar = mean + sd * stats.norm.pdf(Z) / (1.0 - CONF)
    return iar, ciar


# --------------------------------------------------------------------------- #
def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def v1_analytic() -> None:
    section("1. Analytic all-Normal case (deterministic price -> closed form)")
    mu = np.array([1.0, 1.5, 2.0, 0.5, 1.2, 0.8])
    sigma = np.full(6, 2.0)
    spread = np.full(6, 5.0)
    dam = np.full(6, 40.0)
    rep = run_simulation(const_price(spread), normal_imb(mu, sigma), dam_price=dam,
                         config=EngineConfig(n_scenarios=400_000, confidence=CONF, seed=1))

    for name, k, m in (("GROSS", dam + spread, rep.gross), ("SPREAD", spread, rep.spread)):
        a_iar, a_ciar = analytic(k, mu, sigma)
        ok = abs(m.iar - a_iar) / abs(a_iar) < 0.02 and abs(m.ciar - a_ciar) / abs(a_ciar) < 0.02
        print(f"    {name}: IaR engine={m.iar:10,.1f} analytic={a_iar:10,.1f} | "
              f"CIaR engine={m.ciar:10,.1f} analytic={a_ciar:10,.1f}")
        check(f"{name} IaR & CIaR within 2% of closed form", ok)


def v2_convergence() -> None:
    section("2. Convergence: |IaR - analytic| shrinks ~ 1/sqrt(N)")
    mu = np.array([1.0, 2.0, 1.5, 0.5])
    sigma = np.full(4, 2.0)
    spread = np.full(4, 3.0)
    dam = np.full(4, 20.0)
    target, _ = analytic(dam + spread, mu, sigma)

    def avg_err(n: int) -> float:
        errs = [abs(run_simulation(const_price(spread), normal_imb(mu, sigma), dam_price=dam,
                config=EngineConfig(n_scenarios=n, confidence=CONF, seed=s)).gross.iar - target)
                for s in range(8)]
        return float(np.mean(errs))

    ns = [500, 2_000, 8_000, 32_000, 128_000]
    errs = [avg_err(n) for n in ns]
    print(f"    analytic target IaR = {target:,.1f} EUR")
    print(f"    {'N':>8} {'mean |err|':>12} {'err*sqrt(N)':>14}")
    for n, e in zip(ns, errs):
        print(f"    {n:>8,} {e:>12.2f} {e * np.sqrt(n):>14.1f}")
    # error should fall monotonically (allow a tiny wobble) and shrink a lot overall
    monotone = all(errs[i + 1] <= errs[i] * 1.15 for i in range(len(errs) - 1))
    check("error decreases as N grows", monotone)
    check("error at N=128k is < 10% of error at N=500", errs[-1] < 0.1 * errs[0],
          f"{errs[-1]:.2f} vs {errs[0]:.2f}")


def v3_reproducibility() -> None:
    section("3. Seed reproducibility")
    mu, sigma = np.array([1.0, -2.0, 0.5]), np.full(3, 2.0)
    price = QuantilePriceSampler(np.array([0.1, 0.5, 0.9]), np.tile([-5.0, 0.0, 8.0], (3, 1)))
    dam = np.full(3, 30.0)

    a = run_simulation(price, normal_imb(mu, sigma), dam_price=dam,
                       config=EngineConfig(n_scenarios=20_000, seed=99))
    b = run_simulation(price, normal_imb(mu, sigma), dam_price=dam,
                       config=EngineConfig(n_scenarios=20_000, seed=99))
    c = run_simulation(price, normal_imb(mu, sigma), dam_price=dam,
                       config=EngineConfig(n_scenarios=20_000, seed=7))
    print(f"    seed 99 (run A) gross IaR = {a.gross.iar:,.2f}")
    print(f"    seed 99 (run B) gross IaR = {b.gross.iar:,.2f}")
    print(f"    seed  7 (run C) gross IaR = {c.gross.iar:,.2f}")
    check("same seed -> identical IaR", a.gross.iar == b.gross.iar)
    check("different seed -> different IaR (but same ballpark)",
          a.gross.iar != c.gross.iar and abs(a.gross.iar - c.gross.iar) < 0.2 * abs(a.gross.iar))


def v4_summed_quantile() -> None:
    section("4. Summed-quantile (NOT sum of per-MTU IaRs)")
    mu = np.full(6, 1.0)
    sigma = np.full(6, 2.0)
    spread = np.full(6, 4.0)
    dam = np.full(6, 30.0)
    k = dam + spread
    rep = run_simulation(const_price(spread), normal_imb(mu, sigma), dam_price=dam,
                         config=EngineConfig(n_scenarios=400_000, confidence=CONF, seed=2))

    correct = float(np.sum(k * mu)) + Z * float(np.sqrt(np.sum((k * sigma) ** 2)))  # quantile of sum
    naive = float(np.sum(k * mu)) + Z * float(np.sum(np.abs(k) * sigma))            # sum of quantiles
    print(f"    engine (quantile of summed cost) = {rep.gross.iar:10,.1f} EUR")
    print(f"    correct closed form               = {correct:10,.1f} EUR")
    print(f"    WRONG sum-of-per-MTU IaRs          = {naive:10,.1f} EUR")
    print(f"    diversification benefit captured   = {naive - rep.gross.iar:10,.1f} EUR")
    check("engine matches quantile-of-sum (within 2%)", abs(rep.gross.iar - correct) / correct < 0.02)
    check("engine < naive sum-of-per-MTU (no overstatement)", rep.gross.iar < naive)


def main() -> int:
    print("=" * 64)
    print("IaR ENGINE VALIDATION REPORT (Task 2.5)")
    print(f"confidence = {CONF:.0%}   z = {Z:.4f}")
    print("=" * 64)
    v1_analytic()
    v2_convergence()
    v3_reproducibility()
    v4_summed_quantile()
    n_ok = sum(_results)
    n = len(_results)
    print("\n" + "=" * 64)
    print(f"SUMMARY: {n_ok}/{n} checks passed")
    print("=" * 64)
    return 0 if n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
