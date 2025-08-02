# Stage 1: download & quantize the base model
FROM python:3.11-slim AS builder

WORKDIR /tmp/base

# install just what we need for pulling the base model
COPY download_base.py .
RUN pip install --no-cache-dir \
      transformers==4.49.0 \
      bitsandbytes==0.37.0 \
      peft==0.4.0 \
      sentencepiece>=0.1.99
RUN python download_base.py

# Stage 2: the Runpod serverless image
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# copy in the pre-downloaded base
COPY --from=builder /tmp/base /model/base

# install runtime deps, pin bitsandbytes <0.39 and remove any stray triton
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y triton || true

# your inference code + LoRA adapter
COPY inference.py model-inference.tar .
RUN mkdir /model/lora \
 && tar -xzf model-inference.tar -C /model/lora

ENV BASE_MODEL_PATH=/model/base \
    MODEL_DIR=/model/lora \
    HF_MODEL_ID="huggyllama/llama-7b" \
    HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_API_TOKEN}"

CMD ["python", "-u", "inference.py"]
