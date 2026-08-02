#!/bin/sh
set -u

cd /app
if ! python3 /app/packaging/docker/seed_comfy_config.py; then
    echo "[studio] config seeding failed; starting Studio with the existing config."
fi
exec python backend/run.py
