# Multi-zone (country) view

The dashboard aggregates a portfolio's bidding zones into a country-wide view. Zones are
grouped by their area prefix: `SE1`-`SE4` roll up to **Sweden (SE)**, `NO1`-`NO5` to
**Norway (NO)**.

## What you see

The **Overview** tab is the portfolio-wide landing view:

- **Country KPI cards**: the period Gross and Spread IaR for the whole country versus the
  country limit, plus a utilisation bar, a "zones at risk" count and the diversification
  ratio.
- **Per-MBA grid**: one card per bidding zone with its IaR, limit, utilisation bar and a
  status colour (green within limit, amber soft, red breach), so the riskiest zone is obvious
  at a glance. Each card has a **View** button that drills into that zone's Command Centre,
  Risk Analytics and Historical tabs (with a "Country > Zone" breadcrumb).
- **Diversification benefit** chart: the zone IaRs stacked to their naive sum, with a marker
  at the (shorter) diversified country IaR. The gap is the diversification benefit.

## How the country IaR is computed

A country's IaR is **not** the sum of its zones' IaRs - that ignores diversification. Instead
(`iar/risk/aggregate.py`):

1. Each zone's Monte Carlo is run from its current inputs (forecast spread quantiles, DAM
   price, loaded positions), with an **independent** seed per zone.
2. The per-scenario settlement cost is **summed across zones**.
3. The country IaR is the confidence-quantile of that summed cost (CIaR is the tail mean).

So `country IaR = quantile(sum of per-scenario zone cost)`, which is at most the sum of the
zone IaRs - the same summed-quantile logic used for the per-MTU period IaR. The
**diversification ratio** = (sum of zone IaRs) / (country IaR); higher means more benefit.

> Modelling note: zones are drawn **independently**, consistent with the MVP's independence
> assumption. Real bidding-zone imbalances are positively correlated (shared weather), so this
> is diversification-optimistic and biases the country IaR low - the first thing to relax on
> the way to production.

## Service / API

- `iar.service.list_countries()` - countries present, with zone counts.
- `iar.service.list_zones(country)` - the portfolios (zones) in a country.
- `iar.service.get_country_overview(country, confidence=0.95)` - country totals (both bases)
  with limits and severity, plus the per-zone breakdown and the diversification ratios.

## Limits

Country limits live in `config/limits.toml` under `[country.<CODE>]` (with a
`[country.default]` fallback), e.g.:

```toml
[country.SE.gross]
remaining_day_eur = 30000
```

If a country has no entry, its limit falls back to the **sum of its zones' period limits**.

## Keeping zones current

Positions come from windsim (live) or the synthetic generator (demo) and only extend to a
fixed date. The scheduled refresh now covers all zones and logs a warning when any zone's
positions run out within two weeks. Extend them with:

```
python scripts/topup_positions.py            # all areas, extends ~90 days ahead
```

It is idempotent (a fixed per-area seed keeps history stable) and safe to schedule weekly.
