# ─── Stage 1: download & quantize the base model ───
FROM python:3.11-slim AS builder

WORKDIR /tmp/base

# 1) Copy the downloader
COPY download_base.py .

# 2) Install transformers + a new bitsandbytes, then run the script
RUN pip install --no-cache-dir \
      transformers==4.49.0 \
      peft==0.4.0 \
      sentencepiece>=0.1.99 \
      bitsandbytes       && \
    python download_base.py

# ─── Stage 2: your Runpod serverless image ─────────
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# 1) Bring in the pre-quantized base model
COPY --from=builder /tmp/base /model/base

# 2) Install your runtime deps (pin bnb back to 0.37 to avoid Triton issues)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y triton || true

# 3) Copy your inference code & LoRA adapter
COPY inference.py model-inference.tar .
RUN mkdir /model/lora \
 && tar -xzf model-inference.tar -C /model/lora

# 4) Point your code at the two model dirs
ENV BASE_MODEL_PATH=/model/base \
    MODEL_DIR=/model/lora \
    HF_MODEL_ID="huggyllama/llama-7b"

CMD ["python", "-u", "inference.py"]
