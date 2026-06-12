"""Smoke tests for the Task 1.1 skeleton: everything imports cleanly."""

import importlib


def test_package_version():
    import iar

    assert iar.__version__


def test_all_modules_import():
    modules = [
        "iar",
        "iar.db",
        "iar.db.models",
        "iar.db.session",
        "iar.ingestion",
        "iar.ingestion.synthetic",
        "iar.ingestion.clients",
        "iar.ingestion.flatfile_loader",
        "iar.simulation",
        "iar.simulation.imbalance_model",
        "iar.simulation.price_sampler",
        "iar.simulation.engine",
        "iar.risk",
        "iar.risk.alerts",
        "iar.risk.backtest",
        "iar.service",
    ]
    for name in modules:
        importlib.import_module(name)
