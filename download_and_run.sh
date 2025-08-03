#!/usr/bin/env bash
set -euo pipefail

# 1) Ensure MODEL_S3_PATH is provided
if [ -z "${MODEL_S3_PATH:-}" ]; then
  echo "ERROR: MODEL_S3_PATH environment variable is not set!" >&2
  exit 1
fi

echo "⏬ Downloading & extracting model from S3 → ${MODEL_S3_PATH}"
mkdir -p /model

# 2) Split s3://bucket/key into $bucket and $key
#    (strip the "s3://" prefix, then bucket is first segment, key is rest)
s3uri="${MODEL_S3_PATH#s3://}"
bucket="${s3uri%%/*}"
key="${s3uri#*/}"

# 3) Stream from S3 into tar (never write the .tar.gz file itself)
aws s3api get-object \
    --bucket "$bucket" \
    --key "$key" \
    /dev/stdout \
  | tar -xz -C /model

echo "✅ Model extracted into /model"
echo "🚀 Launching inference handler"
exec python -u handler.py
