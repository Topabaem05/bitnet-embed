FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY docs ./docs

RUN pip install --no-cache-dir uv && uv pip install --system .

EXPOSE 8000

CMD ["python", "scripts/run_api.py", "--config", "configs/service/api.yaml"]
