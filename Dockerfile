FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY fraud_explainer ./fraud_explainer
COPY eval ./eval
RUN pip install --no-cache-dir ".[llm]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8001 8002
CMD ["python", "-m", "fraud_explainer", "api"]
