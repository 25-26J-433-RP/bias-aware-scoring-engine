FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential git \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/

RUN python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies from pyproject
RUN pip install --no-cache-dir --prefer-binary fastapi "uvicorn[standard]" pydantic numpy scikit-learn transformers==4.36.2 sentencepiece protobuf \
    && pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

COPY app/ /app/app/
COPY pyproject.toml /app/
COPY README.md /app/
COPY tests/ /app/tests/
COPY pytest.ini /app/

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8004}"]
