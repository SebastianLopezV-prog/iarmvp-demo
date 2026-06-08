# Engine validation (Task 2.5)

How we know the Monte Carlo engine computes the right numbers. Two layers:

- **Automated** — `tests/test_engine.py` (13 tests) asserts the properties below on
  every test run.
- **Demonstrable** — `python scripts/validate_engine.py` prints a readable report of
  the same checks (exit 0 = all passed). Use it to eyeball the evidence.

## The four validations

### 1. Analytic all-Normal case
If the price is made **deterministic**, each scenario's cost `= sum_t k_t * imbalance_t`
is a sum of independent normals, so it is itself Normal with a known mean and standard
deviation. The IaR (a quantile) and CIaR (an Expected Shortfall) then have **closed
forms**:

```
mean = sum_t k_t * mu_t
sd   = sqrt( sum_t (k_t * sigma_t)^2 )
IaR  = mean + z * sd                       z = Phi^-1(confidence)
CIaR = mean + sd * phi(z) / (1 - confidence)
```

The engine matches both to within Monte-Carlo tolerance (< 2%), for Gross (`k = DAM + spread`)
and Spread (`k = spread`).

### 2. Convergence vs scenario count
The estimate error vs the analytic target shrinks roughly as `1/sqrt(N)` — so
`error * sqrt(N)` is approximately constant, and the error at N=128k is < 10% of the
error at N=500. This is the expected Monte-Carlo convergence rate and tells you how many
scenarios buy how much precision.

### 3. Seed reproducibility
Same `seed` -> **identical** IaR (bit-for-bit). Different seeds -> different but
same-ballpark IaR. This is why the persisted run stores its seed: the exact scenario set
can always be regenerated.

### 4. Summed-quantile, NOT sum of per-MTU IaRs
Period IaR is the quantile of the **summed** P&L across MTUs — never the sum of per-MTU
quantiles. The latter assumes every MTU has its worst draw simultaneously and badly
overstates risk. The report shows the engine equals the (correct) quantile-of-sum and is
strictly **below** the naive sum-of-per-MTU number; the gap is the diversification benefit.

## Note on calibration vs correctness
These checks prove the engine is **mathematically correct** given its inputs. They do not
prove the inputs are realistic — in particular the imbalance-uncertainty `sigma` is a
parametric stub today and the DAM spot price is a stub (Gross only). Calibrating `sigma`
against realised outcomes is the Week-3 backtest's job.
