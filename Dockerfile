# API-only mode: ComfyUI and ai-toolkit are host-native and out of scope for this container.
FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt
COPY backend backend
COPY frontend/dist frontend/dist
COPY config.example.json .
COPY packaging/docker/seed_comfy_config.py /app/packaging/docker/seed_comfy_config.py
COPY packaging/docker/studio_api_entrypoint.sh /usr/local/bin/studio-api-entrypoint.sh
RUN chmod 755 /usr/local/bin/studio-api-entrypoint.sh
ENV LDS_DATA_DIR=/data \
    LDS_CONFIG=/data/config.json \
    LDS_HOST=0.0.0.0 \
    LDS_PORT=5050 \
    LDS_AUTO_PORT=0 \
    LDS_DOCKER_COMFY_MODE=none
EXPOSE 5050
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5050/api/health', timeout=3).read()"]
ENTRYPOINT ["/usr/local/bin/studio-api-entrypoint.sh"]
