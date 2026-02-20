
FROM python:3.11-slim
# Set Hugging Face cache directory to a fixed location
ENV TRANSFORMERS_CACHE=/app/hf_cache

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update \
	&& apt-get install -y --no-install-recommends build-essential git \
	&& rm -rf /var/lib/apt/lists/*

COPY pyproject.toml /app/

RUN python -m pip install --upgrade pip setuptools wheel


# Install Python dependencies from pyproject (explicit install to avoid build-system complexity)
# Use --no-cache-dir to keep the image small and prefer binary wheels when available.
RUN pip install --no-cache-dir --prefer-binary \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "pydantic>=2.7.0" \
    "numpy==1.26.4" \
    "scikit-learn==1.4.1.post1" \
    "transformers==4.38.1" \
    "sentencepiece==0.1.99" \
    "protobuf==4.25.3" \
    "pandas>=2.0.0" \
    "requests>=2.31.0" \
    "firebase-admin>=6.5.0" \
    "scipy>=1.11.0" \
    "tqdm>=4.66.0" \
    "slowapi>=0.1.9" \
    && pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY README.md /app/
COPY tests/ /app/tests/
COPY pytest.ini /app/
COPY app/ /app/app/
COPY analysis/ /app/analysis/

# Pre-download models to make the container self-contained
# This prevents "Connection Error" at runtime and reduces cold-start latency
RUN python -m app.download_model

# Force offline mode at runtime to prevent 429 "Too Many Requests" errors
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8004}"]
