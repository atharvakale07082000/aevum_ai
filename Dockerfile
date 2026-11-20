# ---- Build stage ----------------------------------------------------------
FROM python:3.13-slim AS build
WORKDIR /app

# Build the dependency wheels into a venv first, so the layer is cached until
# pyproject.toml changes. The package itself is installed after.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml ./
COPY src/ src/
RUN pip install .

# ---- Runtime stage --------------------------------------------------------
FROM python:3.13-slim AS runtime
WORKDIR /app

# Run as a non-root user.
RUN useradd -r -u 1001 aevum
COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER aevum

EXPOSE 8090

# Probes the port the app actually binds: $PORT when the platform sets it
# (Render does), else 8090. Uses the stdlib so the image needs no curl/wget.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8090'), timeout=2).status == 200 else 1)"]

# Shell form so $PORT expands. settings().port reads AEVUM_AI_PORT, not Render's
# $PORT, so uvicorn is bound to $PORT directly (default 8090 for local docker).
CMD ["sh", "-c", "exec uvicorn aevum_ai.app:app --host 0.0.0.0 --port ${PORT:-8090}"]
