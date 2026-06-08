"""SQLAlchemy ORM models for all IaR entities (Task 1.2).

SQLite is the integration hub: every component reads/writes through these tables
rather than calling each other directly. Models use the SQLAlchemy 2.0 typed
(``Mapped`` / ``mapped_column``) style.

Entity overview
---------------
- ``User`` 1--1 ``Portfolio``; a ``Portfolio`` is the parent of all time-series
  inputs, simulation runs, alerts, and backtest records.
- Time-series tables (positions / forecasts / actuals) are keyed by
  (portfolio_id, timestamp) at MTU resolution.
- Price tables are keyed by (price_area, timestamp) since prices are shared
  across portfolios in the same area.
- ``SimulationRun`` carries a ``vintage_ts`` (as-of time of its inputs) so the
  backtest can join each settled period to the estimate whose vintage precedes it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


# Allowed price areas for the MVP (single area per portfolio).
PRICE_AREAS = ("NO1", "NO2", "SE3")


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)

    portfolio: Mapped["Portfolio"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"

    portfolio_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    price_area: Mapped[str] = mapped_column(String(8), nullable=False)

    user: Mapped["User"] = relationship(back_populates="portfolio")
    dam_positions: Mapped[list["DAMPosition"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    generation_forecasts: Mapped[list["GenerationForecast"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    actual_deliveries: Mapped[list["ActualDelivery"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    simulation_runs: Mapped[list["SimulationRun"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )
    performance_records: Mapped[list["HistoricalPerformanceRecord"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "price_area IN ('NO1', 'NO2', 'SE3')", name="ck_portfolio_price_area"
        ),
    )


# --------------------------------------------------------------------------- #
# Time-series inputs (per portfolio, at MTU resolution)
# --------------------------------------------------------------------------- #
class DAMPosition(Base):
    """Day-ahead market position (MWh sold) per MTU."""

    __tablename__ = "dam_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    mwh: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="dam_positions")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "timestamp", name="uq_dam_pos_pf_ts"),
        Index("ix_dam_pos_pf_ts", "portfolio_id", "timestamp"),
    )


class GenerationForecast(Base):
    """Forecast generation (MWh) per MTU."""

    __tablename__ = "generation_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    forecast_mwh: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="generation_forecasts")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "timestamp", name="uq_gen_fc_pf_ts"),
        Index("ix_gen_fc_pf_ts", "portfolio_id", "timestamp"),
    )


class ActualDelivery(Base):
    """Metered actual generation (MWh) per MTU — used for backtesting."""

    __tablename__ = "actual_deliveries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_mwh: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="actual_deliveries")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "timestamp", name="uq_actual_del_pf_ts"),
        Index("ix_actual_del_pf_ts", "portfolio_id", "timestamp"),
    )


# --------------------------------------------------------------------------- #
# Price series (per price area, shared across portfolios)
# --------------------------------------------------------------------------- #
class ImbalancePriceForecast(Base):
    """Optimeering imbalance price/spread forecast, by area and target timestamp.

    Stored long/tidy: one row per (area, timestamp, statistic, quantile). For a
    quantile statistic, ``quantile`` holds the level (e.g. 10, 25, 50, 75, 90)
    and ``value`` the EUR figure. ``vintage_ts`` is the forecast's event_time,
    enabling vintage replay in the backtest.
    """

    __tablename__ = "imbalance_price_forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price_area: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    vintage_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    statistic_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantile: Mapped[float | None] = mapped_column(Float, nullable=True)
    unit_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(16), nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "price_area",
            "timestamp",
            "vintage_ts",
            "statistic_type",
            "quantile",
            name="uq_imb_fc_key",
        ),
        Index("ix_imb_fc_area_ts", "price_area", "timestamp"),
    )


class DAMPrice(Base):
    """Day-ahead (spot) market price (EUR/MWh) by area and timestamp.

    Source-agnostic: in the MVP this is loaded from a flat file, but the table is
    deliberately keyed only by (price_area, timestamp) so a future external price
    feed (ENTSO-E / Nord Pool / a Volue service) could write to it without any
    change downstream. Required for **Gross IaR**: Optimeering publishes imbalance
    only as a SPREAD vs. spot, so the absolute imbalance price is reconstructed as
    ``imbalance_price = dam_price + spread``.
    """

    __tablename__ = "dam_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price_area: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)  # EUR/MWh

    __table_args__ = (
        UniqueConstraint("price_area", "timestamp", name="uq_dam_price_area_ts"),
        Index("ix_dam_price_area_ts", "price_area", "timestamp"),
    )


class ActualImbalancePrice(Base):
    """Realised imbalance price (EUR/MWh) by area and timestamp — for backtesting."""

    __tablename__ = "actual_imbalance_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    price_area: Mapped[str] = mapped_column(String(8), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint("price_area", "timestamp", name="uq_actual_price_area_ts"),
        Index("ix_actual_price_area_ts", "price_area", "timestamp"),
    )


# --------------------------------------------------------------------------- #
# Simulation runs and results
# --------------------------------------------------------------------------- #
class SimulationRun(Base):
    """One Monte Carlo run for a portfolio, stamped with an input vintage."""

    __tablename__ = "simulation_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    run_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    vintage_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    n_scenarios: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="simulation_runs")
    results: Mapped[list["IaRResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        # Fast backtest lookups: "latest run for a portfolio before a vintage".
        Index("ix_sim_run_pf_vintage", "portfolio_id", "vintage_ts"),
    )


class IaRResult(Base):
    """A single IaR/CIaR figure for a (confidence, horizon, iar_type)."""

    __tablename__ = "iar_results"

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("simulation_runs.run_id"), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    horizon: Mapped[str] = mapped_column(String(32), nullable=False)
    iar_type: Mapped[str] = mapped_column(String(16), nullable=False)
    iar_value: Mapped[float] = mapped_column(Float, nullable=False)
    ciar_value: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["SimulationRun"] = relationship(back_populates="results")
    alerts: Mapped[list["Alert"]] = relationship(back_populates="result")

    __table_args__ = (
        CheckConstraint("iar_type IN ('gross', 'spread')", name="ck_iar_type"),
        Index("ix_iar_result_run", "run_id"),
    )


# --------------------------------------------------------------------------- #
# Alerts and backtesting
# --------------------------------------------------------------------------- #
class Alert(Base):
    """A limit breach raised against an IaRResult."""

    __tablename__ = "alerts"

    alert_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    result_id: Mapped[int] = mapped_column(ForeignKey("iar_results.result_id"), nullable=False)
    limit_type: Mapped[str] = mapped_column(String(32), nullable=False)
    limit_value: Mapped[float] = mapped_column(Float, nullable=False)
    breach_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="alerts")
    result: Mapped["IaRResult"] = relationship(back_populates="alerts")

    __table_args__ = (
        Index("ix_alert_pf", "portfolio_id"),
    )


class HistoricalPerformanceRecord(Base):
    """One backtested period: stored IaR estimate vs realised cost + exceedance."""

    __tablename__ = "historical_performance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.portfolio_id"), nullable=False
    )
    period: Mapped[str] = mapped_column(String(32), nullable=False)
    iar_estimate: Mapped[float] = mapped_column(Float, nullable=False)
    realised_cost: Mapped[float] = mapped_column(Float, nullable=False)
    exceeded: Mapped[bool] = mapped_column(nullable=False)
    kupiec_stat: Mapped[float | None] = mapped_column(Float, nullable=True)

    portfolio: Mapped["Portfolio"] = relationship(back_populates="performance_records")

    __table_args__ = (
        Index("ix_perf_pf_period", "portfolio_id", "period"),
    )
