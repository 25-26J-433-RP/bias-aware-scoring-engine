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

# Install Python dependencies from pyproject (explicit install to avoid build-system complexity)
# Use --no-cache-dir to keep the image small and prefer binary wheels when available.
RUN pip install --no-cache-dir --prefer-binary fastapi "uvicorn[standard]" pydantic numpy scikit-learn

COPY . /app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

