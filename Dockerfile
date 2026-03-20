FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs

RUN pip install --no-cache-dir uv && uv pip install --system .

EXPOSE 8000

CMD ["uvicorn", "bitnet_embed.serve.api:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
