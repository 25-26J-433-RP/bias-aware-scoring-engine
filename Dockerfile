
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
RUN pip install --no-cache-dir --prefer-binary fastapi "uvicorn[standard]" pydantic numpy scikit-learn transformers==4.36.2 sentencepiece protobuf \
    && pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu

COPY app/ /app/app/
COPY pyproject.toml /app/
COPY README.md /app/
COPY tests/ /app/tests/
COPY pytest.ini /app/


# Pre-download Hugging Face model and tokenizer to avoid runtime download/rate limits
# Use TRANSFORMERS_CACHE env var to ensure it persists
# Pre-download Hugging Face model and tokenizer to avoid runtime download/rate limits
# Use TRANSFORMERS_CACHE env var to ensure it persists
ENV PYTHONPATH=/app
RUN HF_HOME=/app/hf_cache TRANSFORMERS_CACHE=/app/hf_cache python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('akura-official/xlm-roberta-large-sinhala-multihead')"
RUN HF_HOME=/app/hf_cache TRANSFORMERS_CACHE=/app/hf_cache python -c "from app.model_multitask_xlmr import SinhalaMultiHeadRegressor; SinhalaMultiHeadRegressor.from_pretrained('akura-official/xlm-roberta-large-sinhala-multihead', trust_remote_code=True)"

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8004}"]
