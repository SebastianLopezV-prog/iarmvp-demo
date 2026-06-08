"""Simulation layer — the calculation core of the PoC.

Modules
-------
- ``imbalance_model`` : expected imbalance (DAM - forecast gen) + uncertainty distribution.
- ``price_sampler``   : inverse-CDF sampling from Optimeering price quantiles.
- ``engine``          : Monte Carlo core; Gross/Spread IaR + CIaR (adapted from the prototype).

MVP assumption: price and imbalance position are sampled INDEPENDENTLY.
"""
