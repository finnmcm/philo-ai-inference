#!/bin/sh
set -e

echo "⏬ Downloading quantized model from S3 → $MODEL_S3_PATH"
mkdir -p /model
aws s3 cp "$MODEL_S3_PATH" /model/model.tar.gz

echo "📦 Extracting model…"
tar -xzf /model/model.tar.gz -C /model

echo "🚀 Starting inference handler"
exec python -u inference.py
