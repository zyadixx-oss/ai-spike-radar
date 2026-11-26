FROM python:3.13-slim
ARG INSTALL_TORCH=false
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$INSTALL_TORCH" = "true" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; fi
COPY app ./app
COPY tests ./tests
COPY pytest.ini .
RUN useradd -m radar && mkdir -p /data && chown radar /data
USER radar
ENV DATABASE_URL=sqlite:////data/spike_radar.db MODELS_DIR=/data/models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/safety', timeout=4)"
CMD ["sh", "-c", "python -m app.cli serve --port ${PORT:-8000}"]
