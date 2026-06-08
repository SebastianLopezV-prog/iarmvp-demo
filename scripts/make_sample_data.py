"""Generate sample/stub flat files for one NO2 wind portfolio (Task 1.4/1.5).

Writes four CSVs to ``data/uploads/`` matching the data contract:
  - dam_positions_NO2.csv      (timestamp, mwh)
  - generation_forecast_NO2.csv (timestamp, forecast_mwh)
  - actual_delivery_NO2.csv     (timestamp, actual_mwh)
  - dam_price_NO2.csv          (timestamp, eur_per_mwh)  -- spot price for Gross IaR

This is synthetic stub data — the plan calls for a "stub portfolio" so the loader
and the end-to-end pipeline can run before any real customer files exist. The
data is deliberately constructed so that ACTUAL generation wobbles around the
FORECAST (a realistic wind forecast error), giving a non-trivial imbalance to
measure. Reproducible via a fixed seed.

Run:  python scripts/make_sample_data.py
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS = PROJECT_ROOT / "data" / "uploads"

# Portfolio assumptions (stub).
CAPACITY_MW = 100.0          # installed wind capacity
MTU_HOURS = 0.25             # 15-minute MTU
N_MTU = 96                   # 24 hours
SEED = 42


def _mtu_index() -> pd.DatetimeIndex:
    """96 fifteen-minute MTU starts beginning at the current UTC midnight."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return pd.DatetimeIndex([start + timedelta(minutes=15 * i) for i in range(N_MTU)])


def generate() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    times = _mtu_index()
    hours = np.array([t.hour + t.minute / 60.0 for t in times])

    # Smooth diurnal wind capacity factor in [0.05, 0.95], plus mild noise.
    factor = 0.45 + 0.25 * np.sin(2 * np.pi * (hours - 3) / 24.0)
    factor = np.clip(factor + rng.normal(0, 0.05, N_MTU), 0.05, 0.95)

    forecast_mwh = CAPACITY_MW * factor * MTU_HOURS           # latest expected generation
    cap_mwh = CAPACITY_MW * MTU_HOURS

    # DAM position was committed day-ahead from an EARLIER forecast, so it differs
    # slightly from the latest generation forecast -> a non-trivial EXPECTED imbalance
    # (DAM - latest forecast). sigma_da ~6% of per-MTU capacity energy.
    sigma_da = 0.06 * cap_mwh
    dam_mwh = np.clip(forecast_mwh + rng.normal(0, sigma_da, N_MTU), 0.0, cap_mwh)

    # Actual = latest forecast + realised forecast error (sigma ~12%); used for backtest.
    sigma = 0.12 * cap_mwh
    actual_mwh = np.clip(forecast_mwh + rng.normal(0, sigma, N_MTU), 0.0, cap_mwh)

    # SYNTHETIC day-ahead (spot) price — OFFLINE FALLBACK ONLY.
    # The REAL NO2 spot now comes from Optimeering's internal MarketsApi (DAM cleared
    # price) via iar/ingestion/markets_client.py; run_pipeline.py / run_iar.py use that
    # and only fall back to this CSV when the internal SDK wheel is unavailable.
    # Smooth diurnal shape + mild noise so the Gross-IaR path still runs offline.
    dam_price = 40.0 + 15.0 * np.sin(2 * np.pi * (hours - 8) / 24.0)
    dam_price = np.maximum(dam_price + rng.normal(0, 3.0, N_MTU), 0.0)

    ts = times.strftime("%Y-%m-%dT%H:%M:%S%z")
    ts = [t[:-2] + ":" + t[-2:] for t in ts]  # 0000 -> 00:00 ISO offset

    return {
        "dam_positions_NO2.csv": pd.DataFrame(
            {"timestamp": ts, "mwh": np.round(dam_mwh, 3)}
        ),
        "generation_forecast_NO2.csv": pd.DataFrame(
            {"timestamp": ts, "forecast_mwh": np.round(forecast_mwh, 3)}
        ),
        "actual_delivery_NO2.csv": pd.DataFrame(
            {"timestamp": ts, "actual_mwh": np.round(actual_mwh, 3)}
        ),
        "dam_price_NO2.csv": pd.DataFrame(
            {"timestamp": ts, "eur_per_mwh": np.round(dam_price, 2)}
        ),
    }


def main() -> None:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    for name, df in generate().items():
        path = UPLOADS / name
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
