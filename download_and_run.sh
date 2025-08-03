#!/bin/sh
set -e

echo "⏬ Downloading model from S3: $MODEL_S3_PATH"
mkdir -p /model
aws s3 cp "$MODEL_S3_PATH" /model/model.tar.gz

echo "📦 Extracting model into /model"
tar -xzf /model/model.tar.gz -C /model

echo "🚀 Launching handler"
exec python -u inference.py
