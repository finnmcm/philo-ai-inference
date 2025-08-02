# 1) BUILD STAGE: download & quantize the base model
FROM python:3.11-slim AS builder

WORKDIR /tmp/base
# install only what’s needed to download & save
RUN pip install --no-cache-dir \
      transformers==4.49.0 \
      bitsandbytes==0.37.0 \
      peft==0.4.0 \
      sentencepiece>=0.1.99

# download & save the tokenizer + 8-bit model
RUN python - << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_8bit=True)
AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=False).save_pretrained("./base")
AutoModelForCausalLM.from_pretrained(
    "huggyllama/llama-7b",
    quantization_config=bnb,
    device_map="cpu"
).save_pretrained("./base")
EOF

# 2) FINAL IMAGE: your runpod serverless handler
FROM runpod/serverless-hello-world:latest

WORKDIR /app

# copy in the pre‐downloaded base model
COPY --from=builder /tmp/base/base /model/base

# install runtime deps, pin bitsandbytes <0.39 to avoid Triton conflicts
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip uninstall -y triton || true

# your code + LoRA adapter
COPY inference.py .
COPY model-inference.tar .
RUN mkdir /model/lora \
 && tar -xzf model-inference.tar -C /model/lora

# tell inference.py where to find things
ENV BASE_MODEL_PATH=/model/base \
    MODEL_DIR=/model/lora

CMD ["python", "-u", "inference.py"]
