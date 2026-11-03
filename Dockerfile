# syntax=docker/dockerfile:1.6
# -----------------------------------------------------------------------------
# Austrian Windatlas — Python 3.12 + R 4.3 + BAMLSS + mgcv
# -----------------------------------------------------------------------------
# A self-contained CPU image for exact reproduction of the paper's
# numbers. GPU users should start from an nvidia/cuda base and install
# the `[gpu]` extra manually.
# -----------------------------------------------------------------------------
FROM mambaorg/micromamba:1.5-bookworm

LABEL org.opencontainers.image.title="austrianwindatlas"
LABEL org.opencontainers.image.description="Observation-constrained Austrian wind climatology — reference implementation"
LABEL org.opencontainers.image.source="https://github.com/ischicker/austrianwindatlas"
LABEL org.opencontainers.image.licenses="MIT"

ARG MAMBA_DOCKERFILE_ACTIVATE=1
USER root

# Minimal OS pieces: build-essential for a handful of R package sources,
# procps for `ps`-based lifecycle tools, git for `pip install -e .`
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        procps \
        git \
 && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER
WORKDIR /opt/austrianwindatlas

# Install Python + R + method deps from a single conda env file (the
# user-facing env from the repo).
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml ./environment.yml
RUN micromamba install -y -n base -f environment.yml \
 && micromamba clean --all --yes

# Install the library itself (editable).  Code is copied last so edits
# below do not invalidate the environment layer.
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml ./pyproject.toml
COPY --chown=$MAMBA_USER:$MAMBA_USER README.md LICENSE NOTICE CITATION.cff ./
COPY --chown=$MAMBA_USER:$MAMBA_USER src ./src
COPY --chown=$MAMBA_USER:$MAMBA_USER R   ./R
COPY --chown=$MAMBA_USER:$MAMBA_USER configs ./configs
COPY --chown=$MAMBA_USER:$MAMBA_USER tests ./tests
COPY --chown=$MAMBA_USER:$MAMBA_USER examples ./examples
RUN pip install --no-deps -e .

# Users mount their data read-only at /data; AWA_DATA_ROOT points there.
ENV AWA_DATA_ROOT=/data
VOLUME ["/data"]

# Smoke test on build so a broken image fails here, not for the user.
RUN python -c "import austrianwindatlas as awa; print(f'austrianwindatlas {awa.__version__} ok')" \
 && Rscript -e 'cat("R", R.version.string, "ok\n"); library(bamlss); library(mgcv)'

CMD ["python", "-c", "import austrianwindatlas as awa; print('run: python examples/<name>.py --help')"]
