ARG pythonversion=3.13

# Build/run image for the Streamlit demo, using uv with the committed lockfile.
FROM python:$pythonversion-slim AS app

RUN set -ex \
    && apt-get update \
    && apt-get install -qq dumb-init \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV LANG="en_US.UTF-8" \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependencies first for better layer caching (uv.lock must be committed: run `uv lock`).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application code, then install the project itself.
COPY . /app
RUN uv sync --frozen --no-dev

EXPOSE 8501
ENTRYPOINT ["dumb-init", "--"]
# All feeds are synthetic; the app self-seeds its database on first load.
CMD ["uv", "run", "streamlit", "run", "app/dashboard.py", \
     "--server.port", "8501", "--server.address", "0.0.0.0", \
     "--server.headless", "true", "--browser.gatherUsageStats", "false"]

