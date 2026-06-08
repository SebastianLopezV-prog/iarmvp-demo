"""Monte Carlo engine (Task 2.3) — adapted from the existing prototype.

Placeholder skeleton (Task 1.1). Will sample independently from the price
marginal and the per-MTU imbalance distribution (>=5,000 scenarios), build
per-scenario P&L, and read IaR off the summed-P&L quantile:

- Gross IaR  = -Q_alpha( sum_t  q_t * lambda_t )
- Spread IaR = -Q_alpha( sum_t  q_t * (lambda_t - p_t) )
- CIaR / Expected Shortfall = mean loss beyond the IaR threshold.

Period IaR is the quantile of the SUMMED P&L across MTUs — never the sum of
per-MTU IaRs. The prototype's t-copula is removed entirely (MVP independence).
"""
