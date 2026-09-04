FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app

COPY pyproject.toml ./
RUN pip install --no-cache-dir numpy pydantic pyyaml fastapi "uvicorn[standard]" pytest httpx

COPY contracts ./contracts
COPY world ./world
COPY brain ./brain
COPY harness ./harness
COPY app ./app
COPY config ./config
COPY scripts ./scripts
COPY tests ./tests

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
