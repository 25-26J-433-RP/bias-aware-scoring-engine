
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
    "fastapi==0.109.2" \
    "uvicorn[standard]==0.27.1" \
    "pydantic==2.6.1" \
    "numpy==1.26.4" \
    "scikit-learn==1.4.1.post1" \
    "transformers==4.38.1" \
    "sentencepiece==0.1.99" \
    "protobuf==4.25.3" \
    && pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

COPY app/ /app/app/
COPY pyproject.toml /app/
ENV HF_HOME=/app/hf_cache
COPY README.md /app/
COPY tests/ /app/tests/
COPY pytest.ini /app/


# Pre-download Hugging Face model and tokenizer to avoid runtime download/rate limits
# Use TRANSFORMERS_CACHE env var to ensure it persists
RUN python -c "from transformers import AutoTokenizer, AutoModel; \
    model_id = 'akura-official/xlm-roberta-large-sinhala-multihead'; \
    AutoTokenizer.from_pretrained(model_id); \
    AutoModel.from_pretrained(model_id, trust_remote_code=True)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8004}"]
