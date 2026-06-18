# Operational tasks

## Running locally

```bash
uv sync --group dev
uv run streamlit run app/dashboard.py
```

On a fresh checkout the database is empty; the app self-seeds the synthetic dataset on first
load (see `app/bootstrap.py`). To rebuild it manually at any time:

```bash
uv run python scripts/seed_synthetic_demo.py --area NO2 --days 30
```

## Hosting (Streamlit Community Cloud)

The demo is hosted on Streamlit Community Cloud, which installs from `requirements.txt` and
runs `app/dashboard.py`. On a cold start the app self-seeds (~1-2 min), then serves and
advances forward while open. No API key or external data is needed.

Because the host is Streamlit Cloud (not Volue's container platform), the cookiecutter's
Docker-build-to-Artifact-Registry workflow is intentionally omitted; a `Dockerfile` is still
provided for local/alternative container runs.

## Docker (optional / alternative)

```bash
uv lock            # required once, commit uv.lock
docker build -t iarmvp-demo .
docker run -p 8501:8501 iarmvp-demo
```

## Backtest / data refresh

`scripts/refresh.py` re-runs the forward IaR, loads settled prices, and re-runs the Kupiec
backtest. The hosted app calls a lightweight version of this on a timer to stay current.
