#!/bin/sh
set -e

echo "⏬ Streaming & extracting quantized model from S3 → $MODEL_S3_PATH"
mkdir -p /model

# Pipe the download directly into tar so we never store the .tar.gz
aws s3 cp "$MODEL_S3_PATH" - | tar -xz -C /model

echo "✅ Model extracted into /model"
echo "🚀 Starting inference handler"
exec python -u inference.py
