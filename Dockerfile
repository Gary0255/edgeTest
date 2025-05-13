FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set up project directory
WORKDIR /app

# Install project dependencies using the lockfile and settings with CPU extras only
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --extra cpu

# Copy the application code
COPY . /app

# Create output directory for logs and results
RUN mkdir -p /app/output && chmod 777 /app/output

# Install the project with its dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra cpu

# Place executables in the environment
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint for direct execution
ENTRYPOINT []

# Run the main application with CPU extras
CMD ["uv", "run", "--extra", "cpu", "main.py"]
