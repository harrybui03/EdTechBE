# Multi-stage build for Agentic RAG API
FROM python:3.12.3-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set environment variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set work directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies and create venv in project directory
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only main && \
    rm -rf $POETRY_CACHE_DIR

# Verify venv was created (debug)
RUN if [ ! -d "/app/.venv" ]; then \
        echo "ERROR: .venv not found!"; \
        poetry env info; \
        exit 1; \
    fi && \
    echo "SUCCESS: .venv found at /app/.venv"

# Production stage
FROM python:3.12.3-slim

# Install system dependencies (minimal for runtime)
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY agentic_rag/ ./agentic_rag/
COPY knowledge-base/ ./knowledge-base/
COPY run_api.py ./

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

# Run the application
CMD ["python", "run_api.py"]

