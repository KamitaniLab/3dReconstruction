FROM rocker/r-ver:4.3.3

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_INSTALL_DIR=/app/.uv-python \
    UV_PYTHON=3.11.10 \
    R_HOME=/usr/local/lib/R \
    LD_LIBRARY_PATH=/usr/local/lib/R/lib \
    PATH="/app/.venv/bin:${PATH}" \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gfortran \
    git \
    libcurl4-openssl-dev \
    libgl1 \
    libglib2.0-0 \
    libicu-dev \
    libssl-dev \
    pkg-config \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo /usr/local/lib/R/lib > /etc/ld.so.conf.d/R.conf && ldconfig

COPY --from=ghcr.io/astral-sh/uv:0.6.16 /uv /uvx /usr/local/bin/

WORKDIR /app/analysis

COPY scripts/install-r-packages.R /app/analysis/scripts/install-r-packages.R
RUN Rscript /app/analysis/scripts/install-r-packages.R

COPY .python-version pyproject.toml uv.lock README.md /app/analysis/
RUN uv sync --locked --no-install-project

COPY src /app/analysis/src
COPY configs /app/analysis/configs
COPY evaluation /app/analysis/evaluation
COPY statistics /app/analysis/statistics
COPY visualization /app/analysis/visualization

RUN uv sync --locked && chmod -R a+rX /app/.venv /app/.uv-python

CMD ["bash"]
