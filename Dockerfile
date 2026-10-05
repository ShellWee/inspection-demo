# syntax=docker/dockerfile:1.7
FROM node:22-bookworm-slim AS frontend-build
WORKDIR /build/frontend
RUN corepack enable
COPY frontend/package.json frontend/pnpm-lock.yaml frontend/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    INSPECTION_DEMO_DATA_DIR=/tmp/inspection-demo/data \
    INSPECTION_DEMO_RUNTIME_WORKSPACE=/tmp/inspection-demo/runtime \
    INSPECTION_DEMO_RESEARCH_DEVICE=cuda \
    INSPECTION_DEMO_FRONTEND_DIST=/app/frontend/dist \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/backend:/app/vendor/cobbie-ecore:/app/vendor/tog-ifc-ecore/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgl1 libgomp1 python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY vendor/tog-ifc-ecore/pyproject.toml vendor/tog-ifc-ecore/README.md ./vendor/tog-ifc-ecore/
COPY vendor/tog-ifc-ecore/src ./vendor/tog-ifc-ecore/src
COPY vendor/text-gnn-plugin ./vendor/text-gnn-plugin
RUN uv pip install --python /app/.venv/bin/python --no-deps ./vendor/tog-ifc-ecore ./vendor/text-gnn-plugin

COPY vendor/cobbie-ecore/src/__init__.py vendor/cobbie-ecore/src/config.py ./vendor/cobbie-ecore/src/
COPY vendor/cobbie-ecore/src/integrations/__init__.py vendor/cobbie-ecore/src/integrations/tog.py vendor/cobbie-ecore/src/integrations/tog_wire.py ./vendor/cobbie-ecore/src/integrations/
COPY vendor/cobbie-ecore/src/baml ./vendor/cobbie-ecore/src/baml
COPY vendor/cobbie-ecore/src/schemas/__init__.py vendor/cobbie-ecore/src/schemas/agent_error.py vendor/cobbie-ecore/src/schemas/result.py ./vendor/cobbie-ecore/src/schemas/
COPY vendor/cobbie-ecore/src/util/__init__.py vendor/cobbie-ecore/src/util/baml_retry.py ./vendor/cobbie-ecore/src/util/
COPY backend/inspection_demo ./backend/inspection_demo
COPY --from=frontend-build /build/frontend/dist ./frontend/dist

RUN if ! getent passwd 1000 >/dev/null; then useradd --create-home --uid 1000 appuser; fi \
    && mkdir -p /tmp/inspection-demo /mnt/default-assets \
    && chown -R 1000:1000 /tmp/inspection-demo /mnt/default-assets
USER 1000
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/v1/health', timeout=3)"
CMD ["python", "-m", "inspection_demo.space_entrypoint"]
