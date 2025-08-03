# Use official PyTorch image with CUDA 11.8 support (A40 → CUDA11.8)
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 1) Install your runtime deps:
#    - runpod for the Runpod API client
#    - transformers pinned to 4.49.0
#    - bitsandbytes-cuda118 for GPU-enabled 4bit/8bit quant
#    - sentencepiece
#    - accelerate & peft for 4-bit + LoRA
#    - awscli to fetch your model from S3
RUN pip install --no-cache-dir \
      runpod \
      transformers==4.49.0 \
      bitsandbytes-cuda118 \
      sentencepiece>=0.1.99 \
      accelerate>=0.22.0 \
      peft>=0.4.0 \
      awscli


# 2) Copy in your code + startup script:
COPY handler.py download_and_run.sh /app/
RUN chmod +x /app/download_and_run.sh

# 3) Document the expected env vars (supplied via runpod.yaml)
ENV AWS_ACCESS_KEY_ID=
ENV AWS_SECRET_ACCESS_KEY=
ENV AWS_DEFAULT_REGION=us-east-1
ENV MODEL_S3_PATH=

# 4) On container start, download+extract model, then exec your handler
ENTRYPOINT ["/app/download_and_run.sh"]
